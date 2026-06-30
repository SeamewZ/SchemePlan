from __future__ import annotations

import html
import json
import gzip
import os
import re
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA_TYPES = [
    "Product",
    "Event",
    "Recipe",
    "Course",
    "Book",
    "Movie",
    "Vehicle",
    "JobPosting",
    "Restaurant",
    "CollegeOrUniversity",
    "LocalBusiness",
    "Organization",
    "SoftwareApplication",
    "Person",
]

COMMON_SCHEMA_PROPS = {
    "@type",
    "@context",
    "additionalType",
    "alternateName",
    "description",
    "identifier",
    "image",
    "mainEntityOfPage",
    "name",
    "sameAs",
    "subjectOf",
    "url",
}

SWDE_VERTICAL_SCHEMA_TYPES = {
    "auto": "Vehicle",
    "book": "Book",
    "camera": "Product",
    "job": "JobPosting",
    "movie": "Movie",
    "nbaplayer": "Person",
    "restaurant": "Restaurant",
    "university": "CollegeOrUniversity",
}

SWDE_ATTRIBUTE_MAP = {
    "auto": {
        "model": "model",
        "price": "offers.price",
        "engine": "vehicleEngine",
        "fuel_economy": "fuelEfficiency",
    },
    "book": {
        "title": "name",
        "author": "author",
        "isbn_13": "isbn",
        "publisher": "publisher",
        "publication_date": "datePublished",
    },
    "camera": {
        "model": "model",
        "price": "offers.price",
        "manufacturer": "manufacturer",
    },
    "job": {
        "title": "title",
        "company": "hiringOrganization",
        "location": "jobLocation",
        "date_posted": "datePosted",
    },
    "movie": {
        "title": "name",
        "director": "director",
        "genre": "genre",
        "mpaa_rating": "contentRating",
    },
    "nbaplayer": {
        "name": "name",
        "team": "memberOf",
        "height": "height",
        "weight": "weight",
    },
    "restaurant": {
        "name": "name",
        "address": "address",
        "phone": "telephone",
        "cuisine": "servesCuisine",
    },
    "university": {
        "name": "name",
        "phone": "telephone",
        "website": "url",
        "type": "additionalType",
    },
}

SWDE_SCHEMA_PROPERTIES: Dict[str, List[str]] = {}
for _vertical, _schema_type in SWDE_VERTICAL_SCHEMA_TYPES.items():
    SWDE_SCHEMA_PROPERTIES.setdefault(_schema_type, [])
    _props = set(SWDE_SCHEMA_PROPERTIES[_schema_type])
    _props.update(COMMON_SCHEMA_PROPS)
    _props.update(field.split(".")[0] for field in SWDE_ATTRIBUTE_MAP[_vertical].values())
    SWDE_SCHEMA_PROPERTIES[_schema_type] = sorted(_props)

SWDE_VERTICALS = sorted(SWDE_VERTICAL_SCHEMA_TYPES)

PLANNER_DEFAULT_MODEL = "gpt-5.5"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


def selected_llm_provider() -> str:
    provider = os.environ.get("SCHEMARAG_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "openai"
    provider = provider.strip().lower()
    if provider in {"gpt", "openai", "openai-responses"}:
        return "openai"
    if provider in {"deepseek", "deepseek-v4-flash", "deepseek-chat"}:
        return "deepseek"
    raise RuntimeError(f"Unsupported SCHEMARAG_LLM_PROVIDER={provider!r}; use 'openai' or 'deepseek'")


def llm_api_key(provider: str) -> str:
    if provider == "deepseek":
        return (
            os.environ.get("SCHEMARAG_DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        )
    return os.environ.get("SCHEMARAG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def llm_model_for_provider(model: Any, provider: str) -> str:
    explicit = str(model or "").strip()
    if provider == "deepseek":
        if not explicit or explicit == PLANNER_DEFAULT_MODEL:
            return os.environ.get("SCHEMARAG_DEEPSEEK_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL
        return explicit
    return explicit or PLANNER_DEFAULT_MODEL


def llm_retry_settings() -> Tuple[int, float]:
    retries = int(os.environ.get("SCHEMARAG_LLM_RETRIES") or os.environ.get("SCHEMARAG_OPENAI_RETRIES", "4"))
    backoff = float(os.environ.get("SCHEMARAG_LLM_RETRY_BACKOFF") or os.environ.get("SCHEMARAG_OPENAI_RETRY_BACKOFF", "2.0"))
    return retries, backoff


def request_json_with_retries(
    url: str,
    body: Dict[str, Any],
    api_key: str,
    timeout: int,
    retries: int,
    backoff: float,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise
        except (URLError, TimeoutError, RemoteDisconnected, IncompleteRead) as exc:
            last_error = exc
            if attempt >= retries:
                raise
        time.sleep(backoff * (2**attempt))
    assert last_error is not None
    raise last_error


def responses_input_to_chat_messages(body: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    chunks.append(str(part.get("text") or part.get("content") or ""))
                else:
                    chunks.append(str(part))
            text = "\n".join(chunk for chunk in chunks if chunk)
        else:
            text = json.dumps(content, ensure_ascii=False)
        messages.append({"role": role, "content": text})
    return messages or [{"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def deepseek_chat_body_from_responses_body(body: Dict[str, Any]) -> Dict[str, Any]:
    thinking_type = os.environ.get("SCHEMARAG_DEEPSEEK_THINKING", "disabled").strip().lower() or "disabled"
    chat_body: Dict[str, Any] = {
        "model": llm_model_for_provider(body.get("model"), "deepseek"),
        "messages": responses_input_to_chat_messages(body),
        "stream": False,
        "thinking": {"type": thinking_type},
    }
    if body.get("max_output_tokens") is not None:
        chat_body["max_tokens"] = int(body["max_output_tokens"])
    if "temperature" in body:
        chat_body["temperature"] = body["temperature"]
    if os.environ.get("SCHEMARAG_DEEPSEEK_JSON_MODE", "1").lower() not in {"0", "false", "no"}:
        chat_body["response_format"] = {"type": "json_object"}
    if thinking_type != "disabled" and os.environ.get("SCHEMARAG_DEEPSEEK_REASONING_EFFORT"):
        chat_body["reasoning_effort"] = os.environ["SCHEMARAG_DEEPSEEK_REASONING_EFFORT"]
    return chat_body


def extract_deepseek_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    if isinstance(data.get("content"), str):
        return str(data["content"])
    return ""


def deepseek_response_text(body: Dict[str, Any], timeout_env: str, default_timeout: str = "120") -> str:
    api_key = llm_api_key("deepseek")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    timeout = int(os.environ.get(timeout_env, default_timeout))
    retries, backoff = llm_retry_settings()
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("SCHEMARAG_DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    url = os.environ.get("SCHEMARAG_DEEPSEEK_CHAT_URL") or f"{base_url.rstrip('/')}/chat/completions"
    data = request_json_with_retries(url, deepseek_chat_body_from_responses_body(body), api_key, timeout, retries, backoff)
    return extract_deepseek_text(data)


def openai_response_text(body: Dict[str, Any], timeout_env: str, default_timeout: str = "120") -> str:
    if selected_llm_provider() == "deepseek":
        return deepseek_response_text(body, timeout_env, default_timeout)
    body = dict(body)
    body["model"] = llm_model_for_provider(body.get("model"), "openai")
    api_key = llm_api_key("openai")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    timeout = int(os.environ.get(timeout_env, default_timeout))
    retries, backoff = llm_retry_settings()
    data = request_json_with_retries("https://api.openai.com/v1/responses", body, api_key, timeout, retries, backoff)
    return extract_openai_text(data)
PLANNER_SLOT_HINTS = {
    "name": {
        "description": "Entity name shown as page title, h1, or explicit name field",
        "evidence_query": ["name", "title", "h1"],
        "value_type": "string",
        "required": True,
    },
    "title": {
        "description": "Job or content title explicitly shown on the page",
        "evidence_query": ["title", "job title", "h1"],
        "value_type": "string",
        "required": True,
    },
    "offers.price": {
        "description": "Displayed price with currency evidence",
        "evidence_query": ["price", "currency", "$", "£", "€", "MSRP", "sale price", "starting price"],
        "value_type": "number/string",
        "required": False,
        "normalization": "remove currency symbol only when a downstream numeric field requires it",
    },
    "offers.priceCurrency": {
        "description": "Displayed or implied currency for the listed offer price",
        "evidence_query": ["currency", "$", "£", "€", "USD", "GBP", "EUR"],
        "value_type": "string",
        "required": False,
    },
    "offers.availability": {
        "description": "Product availability or stock state explicitly shown on the page",
        "evidence_query": ["availability", "stock", "in stock", "out of stock"],
        "value_type": "string",
        "required": False,
    },
    "sku": {
        "description": "SKU or UPC identifier explicitly shown in product information",
        "evidence_query": ["sku", "upc", "product information"],
        "value_type": "string",
        "required": False,
    },
    "mpn": {
        "description": "Manufacturer part number, stock number, or product code explicitly shown in the page",
        "evidence_query": ["mpn", "manufacturer stock number", "part number", "model number", "product code", "title"],
        "value_type": "string",
        "required": False,
    },
    "brand": {
        "description": "Product brand explicitly shown in the title, heading, or product details",
        "evidence_query": ["brand", "maker", "make", "manufacturer", "title", "h1"],
        "value_type": "string",
        "required": False,
    },
    "category": {
        "description": "Product category or product type explicitly shown in details, breadcrumbs, title, or description",
        "evidence_query": ["category", "product type", "type", "breadcrumb", "department"],
        "value_type": "string",
        "required": False,
    },
    "material": {
        "description": "Product material explicitly shown in title, details, or description",
        "evidence_query": ["material", "metal type", "fabric", "steel", "wood", "plastic"],
        "value_type": "string",
        "required": False,
    },
    "width": {
        "description": "Product width or width-like dimension explicitly shown in details or dimension text",
        "evidence_query": ["width", "w", "dimensions", "size", "inches"],
        "value_type": "string",
        "required": False,
    },
    "depth": {
        "description": "Product depth or length dimension explicitly shown in details or description",
        "evidence_query": ["depth", "length", "deep", "d", "dimensions", "inches"],
        "value_type": "string",
        "required": False,
    },
    "size": {
        "description": "Product size, capacity, pack quantity, or count explicitly shown in details or description",
        "evidence_query": ["size", "capacity", "pack quantity", "count", "carton", "dimensions"],
        "value_type": "string",
        "required": False,
    },
    "aggregateRating.ratingValue": {
        "description": "Displayed rating value or star-rating class evidence",
        "evidence_query": ["rating", "star", "stars", "star-rating"],
        "value_type": "number/string",
        "required": False,
    },
    "aggregateRating.reviewCount": {
        "description": "Displayed review count",
        "evidence_query": ["review count", "number of reviews", "reviews"],
        "value_type": "number/string",
        "required": False,
    },
    "author": {
        "description": "Book author explicitly shown on the page",
        "evidence_query": ["author", "by"],
        "value_type": "string/list",
        "required": False,
    },
    "publisher": {
        "description": "Book publisher explicitly shown on the page",
        "evidence_query": ["publisher"],
        "value_type": "string",
        "required": False,
    },
    "isbn": {
        "description": "ISBN identifier explicitly shown on the page",
        "evidence_query": ["isbn", "isbn-13"],
        "value_type": "string",
        "required": False,
    },
    "datePublished": {
        "description": "Publication date explicitly shown on the page",
        "evidence_query": ["published", "publication date", "pub date", "release date"],
        "value_type": "date/string",
        "required": False,
    },
    "model": {
        "description": "Product or vehicle model explicitly shown on the page",
        "evidence_query": ["model", "h1", "title", "trim", "modelname", "model_vch"],
        "value_type": "string",
        "required": False,
    },
    "manufacturer": {
        "description": "Product manufacturer explicitly shown on the page",
        "evidence_query": ["manufacturer", "maker", "brand", "make", "mfr"],
        "value_type": "string",
        "required": False,
    },
    "color": {
        "description": "Product color explicitly shown in product details or description text",
        "evidence_query": ["color", "colour", "color(s)", "product details"],
        "value_type": "string",
        "required": False,
    },
    "vehicleEngine": {
        "description": "Vehicle engine description explicitly shown on the page",
        "evidence_query": ["engine"],
        "value_type": "string",
        "required": False,
    },
    "fuelEfficiency": {
        "description": "Fuel economy or MPG explicitly shown on the page",
        "evidence_query": ["fuel economy", "mpg"],
        "value_type": "string",
        "required": False,
    },
    "hiringOrganization": {
        "description": "Hiring company or employer explicitly shown on the page",
        "evidence_query": ["company", "employer", "organization"],
        "value_type": "string",
        "required": False,
    },
    "jobLocation": {
        "description": "Job location explicitly shown on the page",
        "evidence_query": ["location", "city", "state"],
        "value_type": "string",
        "required": False,
    },
    "datePosted": {
        "description": "Job posting date explicitly shown on the page",
        "evidence_query": ["date posted", "posted"],
        "value_type": "date/string",
        "required": False,
    },
    "director": {
        "description": "Movie director explicitly shown on the page",
        "evidence_query": ["director"],
        "value_type": "string/list",
        "required": False,
    },
    "genre": {
        "description": "Movie genre or category explicitly shown on the page",
        "evidence_query": ["genre"],
        "value_type": "string/list",
        "required": False,
    },
    "contentRating": {
        "description": "MPAA or content rating explicitly shown on the page",
        "evidence_query": ["mpaa", "rating", "content rating"],
        "value_type": "string",
        "required": False,
    },
    "memberOf": {
        "description": "Team or organization membership explicitly shown on the page",
        "evidence_query": ["team"],
        "value_type": "string",
        "required": False,
    },
    "height": {
        "description": "Height explicitly shown on the page as a person height or product dimension",
        "evidence_query": ["height", "h", "dimensions", "inches"],
        "value_type": "string",
        "required": False,
    },
    "weight": {
        "description": "Person weight explicitly shown on the page",
        "evidence_query": ["weight"],
        "value_type": "string",
        "required": False,
    },
    "address": {
        "description": "Street or postal address explicitly shown on the page",
        "evidence_query": ["address", "street", "city", "state", "zip"],
        "value_type": "string",
        "required": False,
    },
    "telephone": {
        "description": "Phone number explicitly shown on the page",
        "evidence_query": ["phone", "telephone"],
        "value_type": "string",
        "required": False,
    },
    "servesCuisine": {
        "description": "Cuisine type explicitly shown on the page",
        "evidence_query": ["cuisine"],
        "value_type": "string/list",
        "required": False,
    },
    "url": {
        "description": "Official website URL explicitly shown on the page or base tag",
        "evidence_query": ["website", "url", "link"],
        "value_type": "url/string",
        "required": False,
    },
    "additionalType": {
        "description": "Explicit category or type label shown on the page",
        "evidence_query": ["type", "category", "type of school", "private", "public"],
        "value_type": "string/url",
        "required": False,
    },
}

FIELD_ALIASES = {
    "name": ["name"],
    "title": ["title", "job title"],
    "description": ["description", "summary", "about"],
    "url": ["url", "website", "link"],
    "image": ["image", "photo", "thumbnail", "poster"],
    "sku": ["sku"],
    "mpn": ["mpn", "model"],
    "model": ["model"],
    "brand": ["brand", "maker", "make"],
    "manufacturer": ["manufacturer", "mfr", "brand", "make"],
    "category": ["category", "product type", "type", "department"],
    "color": ["color", "colour", "color(s)", "colors"],
    "material": ["material", "metal type", "fabric"],
    "width": ["width", "w"],
    "depth": ["depth", "length", "deep", "d"],
    "size": ["size", "capacity", "pack quantity", "count"],
    "availability": ["availability", "stock"],
    "price": ["price", "cost", "msrp", "starting msrp", "sale price", "market price"],
    "priceCurrency": ["currency"],
    "ratingValue": ["rating", "star rating", "stars"],
    "reviewCount": ["review count", "number of reviews", "reviews"],
    "startDate": ["start date", "starts", "date"],
    "endDate": ["end date", "ends"],
    "location": ["location", "venue", "address"],
    "jobLocation": ["job location", "location"],
    "hiringOrganization": ["company", "employer", "organization"],
    "datePosted": ["date posted", "posted", "posting date"],
    "performer": ["performer", "artist"],
    "author": ["author", "by"],
    "isbn": ["isbn"],
    "datePublished": ["published", "publication date", "pub date", "release date"],
    "vehicleEngine": ["engine"],
    "fuelEfficiency": ["fuel economy", "mpg"],
    "director": ["director"],
    "contentRating": ["mpaa rating", "mpaa", "content rating"],
    "memberOf": ["team"],
    "height": ["height"],
    "weight": ["weight"],
    "servesCuisine": ["cuisine"],
    "additionalType": ["type"],
    "recipeIngredient": ["ingredient", "ingredients"],
    "recipeInstructions": ["instruction", "directions", "steps"],
    "cookTime": ["cook time"],
    "prepTime": ["prep time"],
    "telephone": ["phone", "telephone", "tel"],
    "email": ["email"],
    "address": ["address", "street", "city", "state", "zip"],
    "applicationCategory": ["category"],
    "operatingSystem": ["operating system", "os"],
    "aggregateRating": ["rating", "stars"],
    "courseCode": ["course code"],
    "provider": ["provider", "university", "school"],
    "publisher": ["publisher"],
    "genre": ["genre"],
    "inLanguage": ["language"],
}

PROPERTY_EVIDENCE_TERMS = {
    "offers.price": ["price", "currency", "$", "£", "€", "usd", "gbp", "eur", "cost", "msrp", "starting msrp", "market price", "sale price"],
    "offers.priceCurrency": ["currency", "$", "£", "€", "usd", "gbp", "eur"],
    "offers.availability": ["availability", "stock", "in stock", "out of stock"],
    "aggregateRating.ratingValue": ["rating", "star", "stars", "star-rating", "aggregate rating"],
    "aggregateRating.reviewCount": ["review count", "number of reviews", "reviews", "customer reviews"],
    "sku": ["sku", "upc", "product information", "product details"],
    "mpn": ["mpn", "model", "product information", "product details"],
    "isbn": ["isbn", "isbn-13", "isbn 13"],
    "telephone": ["phone", "telephone", "tel"],
    "url": ["website", "url", "link", "href"],
    "model": ["model", "modelname", "model_vch", "trim", "h1", "title"],
    "datePublished": ["published", "publication date", "pub date", "release date"],
    "datePosted": ["date posted", "posted", "posting date"],
    "address": ["address", "street", "city", "state", "zip"],
    "manufacturer": ["manufacturer", "mfr", "brand", "make"],
    "brand": ["brand", "manufacturer", "maker", "make", "h1", "title"],
    "category": ["category", "product type", "type", "department", "breadcrumb"],
    "material": ["material", "metal type", "fabric", "steel", "wood", "plastic", "metal"],
    "width": ["width", " w ", "dimensions", "inches", "\"w", "wide"],
    "height": ["height", " h ", "dimensions", "inches", "\"h", "tall"],
    "depth": ["depth", "length", "deep", " d ", "dimensions", "inches"],
    "size": ["size", "capacity", "pack quantity", "count", "carton", "pack", "dimensions"],
    "color": ["color", "colour", "color(s)", "colors", "product details"],
}

ATTRIBUTE_EVIDENCE_CUES = {
    "price",
    "currency",
    "rating",
    "star",
    "review",
    "sku",
    "upc",
    "isbn",
    "model",
    "manufacturer",
    "color",
    "author",
    "publisher",
    "phone",
    "telephone",
    "address",
    "cuisine",
    "director",
    "genre",
    "availability",
    "title",
    "name",
}

SOURCE_SUPPORTED_PREFIX = "__SOURCE_SUPPORTED__:"

RICH_PRODUCT_SLOT_PATHS = {
    "description",
    "image",
    "url",
    "brand",
    "color",
    "mpn",
    "gtin",
    "gtin8",
    "gtin12",
    "gtin13",
    "gtin14",
    "offers.priceValidUntil",
    "offers.itemCondition",
    "offers.seller",
    "offers.hasMerchantReturnPolicy",
    "offers.priceSpecification",
}

STRICT_IDENTITY_EVIDENCE_PATHS = {"manufacturer", "model", "brand", "mpn"}

PRODUCT_CONTEXT_FALLBACK_PATHS = {
    "brand",
    "category",
    "color",
    "depth",
    "height",
    "manufacturer",
    "material",
    "model",
    "mpn",
    "size",
    "sku",
    "width",
}


def fetch_url(url: str, timeout: int = 25) -> Tuple[bool, str, str]:
    req = Request(url, headers={"User-Agent": "WISE-SchemaRAG/0.1"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return True, raw.decode("utf-8", errors="replace"), ""
    except Exception as exc:  # network failures must be logged, not hidden
        return False, "", f"{type(exc).__name__}: {exc}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_tags(fragment: str) -> str:
    fragment = html.unescape(fragment)
    fragment = re.sub(r"(?is)<script.*?</script>", " ", fragment)
    fragment = re.sub(r"(?is)<style.*?</style>", " ", fragment)
    fragment = re.sub(r"(?is)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?is)</p>|</div>|</li>|</h\d>", "\n", fragment)
    fragment = re.sub(r"(?is)<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment)
    lines = [re.sub(r"\s+", " ", line).strip() for line in fragment.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_json_from_script_pre(pre_text: str) -> Any | None:
    decoded = html.unescape(pre_text).strip()
    decoded = re.sub(r"(?is)^<script[^>]*>", "", decoded)
    decoded = re.sub(r"(?is)</script>\s*$", "", decoded).strip()
    if not decoded or not decoded.startswith(("{", "[")):
        return None
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return None


def find_primary_object(obj: Any, schema_type: str) -> Dict[str, Any] | None:
    if isinstance(obj, list):
        for item in obj:
            found = find_primary_object(item, schema_type)
            if found:
                return found
        return obj[0] if obj and isinstance(obj[0], dict) else None
    if not isinstance(obj, dict):
        return None
    graph = obj.get("@graph")
    if isinstance(graph, list):
        for item in graph:
            if isinstance(item, dict) and type_matches(item.get("@type"), schema_type):
                return item
        for item in graph:
            if isinstance(item, dict):
                return item
    return obj


def type_matches(value: Any, schema_type: str) -> bool:
    if isinstance(value, list):
        return any(type_matches(v, schema_type) for v in value)
    return str(value).split(":")[-1] == schema_type


def extract_schema_properties(schema_html: str) -> List[str]:
    props = set(COMMON_SCHEMA_PROPS)
    for match in re.finditer(
        r'(?is)<th class="prop-nam"><code>.*?<a href="/([^"]+)"[^>]*>(.*?)</a>',
        schema_html,
    ):
        prop = strip_tags(match.group(2)).strip()
        if prop:
            props.add(prop)
    return sorted(props)


def extract_examples_from_schema_page(schema_type: str, schema_html: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    blocks = re.split(r'(?is)<div class="example-head">', schema_html)[1:]
    for idx, block in enumerate(blocks, 1):
        pre_blocks = re.findall(r"(?is)<pre[^>]*>(.*?)</pre>", block)
        if not pre_blocks:
            continue
        source_text = strip_tags(pre_blocks[0])
        json_obj = None
        for pre in pre_blocks:
            if "application/ld+json" in html.unescape(pre) or '"@type"' in html.unescape(pre):
                json_obj = extract_json_from_script_pre(pre)
                if json_obj is not None:
                    break
        primary = find_primary_object(json_obj, schema_type) if json_obj is not None else None
        if not primary:
            continue
        fields = flatten_fields(primary)
        if len([k for k in fields if not k.startswith("@")]) < 2:
            continue
        examples.append(
            {
                "id": f"{schema_type.lower()}_{idx:03d}",
                "schema_type": schema_type,
                "source": "schema.org examples",
                "source_url": f"https://schema.org/{schema_type}",
                "text": source_text or make_text_from_json(primary),
                "gold": primary,
            }
        )
    return examples


class DomNode:
    def __init__(
        self,
        node_id: str,
        tag: str,
        attrs: Dict[str, str],
        parent: Optional["DomNode"],
        sibling_index: int,
    ) -> None:
        self.node_id = node_id
        self.tag = tag.lower()
        self.attrs = attrs
        self.parent = parent
        self.children: List["DomNode"] = []
        self.text_parts: List[str] = []
        self.sibling_index = sibling_index


class EvidenceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = DomNode("dom_0000", "document", {}, None, 1)
        self.stack = [self.root]
        self.count = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        parent = self.stack[-1]
        self.count += 1
        attr_map = {name.lower(): value or "" for name, value in attrs}
        sibling_index = sum(1 for child in parent.children if child.tag == tag.lower()) + 1
        node = DomNode(f"dom_{self.count:04d}", tag, attr_map, parent, sibling_index)
        parent.children.append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].text_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.stack[-1].text_parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.stack[-1].text_parts.append(f"&#{name};")


def build_dom_evidence_graph(page: str, max_blocks: int = 160) -> Dict[str, Any]:
    parser = EvidenceHTMLParser()
    parser.feed(page or "")
    blocks: List[Dict[str, Any]] = []
    dom_to_evidence: Dict[str, str] = {}
    all_nodes = list(iter_dom_nodes(parser.root))
    evidence_nodes = [node for node in all_nodes if is_evidence_node(node)]
    for idx, node in enumerate(evidence_nodes[:max_blocks], 1):
        evidence_id = f"ev_{idx:04d}"
        dom_to_evidence[node.node_id] = evidence_id
        blocks.append(make_evidence_block(evidence_id, node))
    blocks.extend(source_evidence_blocks_from_html(page or "", len(blocks)))
    edges = make_evidence_edges(evidence_nodes[:max_blocks], dom_to_evidence)
    summary = {
        "block_count": len(blocks),
        "tags": sorted({block["html_tag"] for block in blocks}),
        "source_kinds": sorted({block["source_kind"] for block in blocks}),
        "page_title": first_block_text(blocks, "title"),
        "source_derived_paths": sorted(
            {
                str(block.get("attribute_cues", {}).get("source_path"))
                for block in blocks
                if block.get("attribute_cues", {}).get("source_path")
            }
        )[:80],
    }
    return {"blocks": blocks, "edges": edges, "summary": summary}


def source_evidence_blocks_from_html(page: str, start_index: int = 0) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    next_index = start_index
    for script_idx, match in enumerate(
        re.finditer(r"(?is)<script[^>]+type=[\"'][^\"']*ld\+json[^\"']*[\"'][^>]*>(.*?)</script>", page or ""),
        start=1,
    ):
        obj = json_loads_loose(match.group(1))
        if obj is None:
            continue
        for source_path, value in flatten_source_json_values(obj):
            slot_path = source_path_to_slot_path(source_path)
            if not slot_path or not str(value).strip():
                continue
            next_index += 1
            blocks.append(make_source_evidence_block(next_index, "script", slot_path, str(value), f"/source-jsonld[{script_idx}]/{source_path}", source_path))

    attr_re = re.compile(r"(?is)<(?P<tag>meta|link|img|a|span|div|time|data|button|input)\b(?P<attrs>[^>]*)>")
    pair_re = re.compile(r"([a-zA-Z_:.-]+)\s*=\s*(['\"])(.*?)\2", re.S)
    for attr_idx, match in enumerate(attr_re.finditer(page or ""), start=1):
        attrs = {m.group(1).lower(): html.unescape(m.group(3)) for m in pair_re.finditer(match.group("attrs"))}
        raw_key = (attrs.get("itemprop") or attrs.get("property") or attrs.get("name") or attrs.get("rel") or "").strip()
        if not raw_key:
            continue
        slot_path = META_PROPERTY_PATH_MAP.get(raw_key.lower()) or source_path_to_slot_path(raw_key)
        value = attrs.get("content") or attrs.get("href") or attrs.get("src") or attrs.get("alt") or attrs.get("value") or ""
        if not slot_path or not value.strip():
            continue
        next_index += 1
        block = make_source_evidence_block(next_index, match.group("tag").lower(), slot_path, value, f"/source-attr[{attr_idx}]", raw_key)
        block["attribute_cues"]["source_key"] = raw_key
        blocks.append(block)
    return blocks


META_PROPERTY_PATH_MAP = {
    "og:title": "name",
    "twitter:title": "name",
    "og:description": "description",
    "twitter:description": "description",
    "description": "description",
    "og:image": "image",
    "twitter:image": "image",
    "og:url": "url",
    "canonical": "url",
    "product:price:amount": "offers.price",
    "product:price:currency": "offers.priceCurrency",
    "product:availability": "offers.availability",
    "price": "offers.price",
    "availability": "offers.availability",
    "sku": "sku",
    "mpn": "mpn",
    "brand": "brand",
}


def make_source_evidence_block(index: int, tag: str, slot_path: str, value: str, xpath: str, source_path: str) -> Dict[str, Any]:
    evidence_id = f"src_{index:04d}"
    max_chars = int(os.environ.get("SOURCE_EVIDENCE_TEXT_LIMIT", "1800") or "1800")
    return {
        "evidence_id": evidence_id,
        "text": f"{slot_path}: {str(value).strip()[:max_chars]}",
        "html_tag": tag,
        "xpath": xpath,
        "css_path": f"{tag}[source]::{slot_path}",
        "dom_depth": 1,
        "parent_id": None,
        "child_ids": [],
        "sibling_ids": [],
        "previous_sibling_id": None,
        "next_sibling_id": None,
        "relations": {"parent": None, "children": [], "siblings": []},
        "table_context": {"in_table": False, "headers": [], "row_header": "", "table_xpath": ""},
        "attribute_cues": {"source_path": slot_path, "jsonld_path": source_path},
        "source_kind": "structured-jsonld" if tag == "script" else "source-attribute",
    }


def json_loads_loose(text: str) -> Any | None:
    raw = html.unescape(text).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def flatten_source_json_values(obj: Any, prefix: str = "", depth: int = 0) -> Iterable[Tuple[str, str]]:
    if depth > 8:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "@context":
                continue
            clean_key = key.split(":")[-1]
            path = f"{prefix}.{clean_key}" if prefix else clean_key
            if isinstance(value, (str, int, float, bool)):
                yield path, str(value)
            elif isinstance(value, dict):
                yield from flatten_source_json_values(value, path, depth + 1)
            elif isinstance(value, list):
                for idx, item in enumerate(value[:8]):
                    if isinstance(item, (str, int, float, bool)):
                        yield path, str(item)
                    else:
                        yield from flatten_source_json_values(item, f"{path}.{idx}", depth + 1)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj[:8]):
            yield from flatten_source_json_values(item, f"{prefix}.{idx}" if prefix else str(idx), depth + 1)


def source_path_to_slot_path(path: str) -> str:
    clean = re.sub(r"^\d+\.", "", str(path)).replace("@", "")
    clean = clean.replace("mainEntity.", "")
    parts = [part for part in clean.split(".") if part and part not in {"context", "type", "graph"}]
    if not parts:
        return ""
    if len(parts) >= 2 and parts[-2] in {"offers", "aggregateRating", "hasMerchantReturnPolicy", "priceSpecification"}:
        return ".".join(parts[-2:])
    leaf = parts[-1]
    return {
        "price": "offers.price",
        "priceCurrency": "offers.priceCurrency",
        "availability": "offers.availability",
        "itemCondition": "offers.itemCondition",
        "priceValidUntil": "offers.priceValidUntil",
        "seller": "offers.seller",
        "ratingValue": "aggregateRating.ratingValue",
        "reviewCount": "aggregateRating.reviewCount",
    }.get(leaf, leaf)


def iter_dom_nodes(node: DomNode) -> Iterable[DomNode]:
    yield node
    for child in node.children:
        yield from iter_dom_nodes(child)


def evidence_text_for_node(node: DomNode) -> str:
    max_chars = int(os.environ.get("DOM_EVIDENCE_TEXT_LIMIT", "500") or "500")
    if node.tag == "tr":
        cells = [evidence_text_for_node(child) for child in node.children if child.tag in {"th", "td"}]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            return f"{cells[0]}: {' | '.join(cells[1:])}"[:max_chars]
        if cells:
            return " | ".join(cells)[:max_chars]
    raw = "".join(node.text_parts)
    if node.tag in {"script", "style"} and "ld+json" not in node.attrs.get("type", "").lower():
        return ""
    text = strip_tags(raw)
    if node.tag == "script" and "ld+json" in node.attrs.get("type", "").lower():
        text = html.unescape(raw).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def attribute_evidence_text(node: DomNode) -> str:
    cue = dom_cue_text(node)
    if cue:
        return cue
    parts = []
    for key in ("itemprop", "property", "typeof", "id", "class", "name", "aria-label", "role"):
        value = node.attrs.get(key, "")
        if not value:
            continue
        norm_value = normalize_text(value)
        if key in {"itemprop", "property"} or any(cue in norm_value for cue in ATTRIBUTE_EVIDENCE_CUES):
            parts.append(f"{key}: {value}")
    return "; ".join(parts)[:240]


def dom_cue_text(node: DomNode) -> str:
    classes = split_classes(node.attrs.get("class", ""))
    if any(cls.lower() == "star-rating" for cls in classes):
        rating = star_rating_value(" ".join(classes))
        if rating:
            return f"Star rating: {rating} stars"
    return ""


def is_evidence_node(node: DomNode) -> bool:
    return bool(evidence_text_for_node(node) or attribute_evidence_text(node))


def make_evidence_block(evidence_id: str, node: DomNode) -> Dict[str, Any]:
    parent = node.parent
    children_with_text = [child for child in node.children if is_evidence_node(child)]
    siblings = [sibling for sibling in parent.children if sibling is not node] if parent else []
    text_siblings = [sibling for sibling in siblings if is_evidence_node(sibling)]
    prev_sibling, next_sibling = adjacent_siblings(node)
    return {
        "evidence_id": evidence_id,
        "text": evidence_text_for_node(node) or attribute_evidence_text(node),
        "html_tag": node.tag,
        "xpath": xpath_for_node(node),
        "css_path": css_path_for_node(node),
        "dom_depth": dom_depth(node),
        "parent_id": parent.node_id if parent and parent.tag != "document" else None,
        "child_ids": [child.node_id for child in children_with_text],
        "sibling_ids": [sibling.node_id for sibling in text_siblings],
        "previous_sibling_id": prev_sibling.node_id if prev_sibling else None,
        "next_sibling_id": next_sibling.node_id if next_sibling else None,
        "relations": {
            "parent": parent.node_id if parent and parent.tag != "document" else None,
            "children": [child.node_id for child in children_with_text],
            "siblings": [sibling.node_id for sibling in text_siblings],
        },
        "table_context": table_context_for_node(node),
        "attribute_cues": attribute_cues(node),
        "source_kind": source_kind(node),
    }


def make_evidence_edges(nodes: List[DomNode], dom_to_evidence: Dict[str, str]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    node_set = {node.node_id for node in nodes}
    for node in nodes:
        src = dom_to_evidence[node.node_id]
        if node.parent and node.parent.node_id in node_set:
            edges.append({"source": dom_to_evidence[node.parent.node_id], "target": src, "relation": "parent"})
        elif node.parent and node.parent.tag != "document":
            edges.append({"source": node.parent.node_id, "target": src, "relation": "parent"})
        prev_sibling, next_sibling = adjacent_siblings(node)
        if prev_sibling and prev_sibling.node_id in node_set:
            edges.append({"source": dom_to_evidence[prev_sibling.node_id], "target": src, "relation": "previous_sibling"})
        if next_sibling and next_sibling.node_id in node_set:
            edges.append({"source": src, "target": dom_to_evidence[next_sibling.node_id], "relation": "next_sibling"})
    return edges


def adjacent_siblings(node: DomNode) -> Tuple[Optional[DomNode], Optional[DomNode]]:
    if not node.parent:
        return None, None
    siblings = [child for child in node.parent.children if is_evidence_node(child)]
    try:
        idx = siblings.index(node)
    except ValueError:
        return None, None
    prev_sibling = siblings[idx - 1] if idx > 0 else None
    next_sibling = siblings[idx + 1] if idx + 1 < len(siblings) else None
    return prev_sibling, next_sibling


def xpath_for_node(node: DomNode) -> str:
    parts = []
    cur: Optional[DomNode] = node
    while cur and cur.tag != "document":
        parts.append(f"{cur.tag}[{cur.sibling_index}]")
        cur = cur.parent
    return "/" + "/".join(reversed(parts))


def css_path_for_node(node: DomNode) -> str:
    parts = []
    cur: Optional[DomNode] = node
    while cur and cur.tag != "document":
        part = cur.tag
        if cur.attrs.get("id"):
            part += f"#{sanitize_css_token(cur.attrs['id'])}"
        classes = split_classes(cur.attrs.get("class", ""))
        if classes:
            part += "".join(f".{sanitize_css_token(cls)}" for cls in classes[:3])
        parts.append(part)
        cur = cur.parent
    return " > ".join(reversed(parts))


def sanitize_css_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())[:80] or "x"


def split_classes(value: str) -> List[str]:
    return [part for part in re.split(r"\s+", value.strip()) if part]


def dom_depth(node: DomNode) -> int:
    depth = 0
    cur = node.parent
    while cur:
        if cur.tag != "document":
            depth += 1
        cur = cur.parent
    return depth


def table_context_for_node(node: DomNode) -> Dict[str, Any]:
    row = nearest_ancestor(node, "tr")
    table = nearest_ancestor(node, "table")
    headers: List[str] = []
    row_header = ""
    if row:
        for child in row.children:
            if child is node:
                break
            if child.tag == "th":
                row_header = evidence_text_for_node(child)
                if row_header:
                    headers.append(row_header)
        if not headers:
            headers.extend(evidence_text_for_node(child) for child in row.children if child.tag == "th")
            headers = [header for header in headers if header]
    return {
        "in_table": table is not None,
        "headers": headers[:8],
        "row_header": row_header,
        "table_xpath": xpath_for_node(table) if table else "",
    }


def nearest_ancestor(node: Optional[DomNode], tag: str) -> Optional[DomNode]:
    cur = node
    while cur:
        if cur.tag == tag:
            return cur
        cur = cur.parent
    return None


def attribute_cues(node: DomNode) -> Dict[str, Any]:
    cue_keys = [
        "id",
        "class",
        "itemprop",
        "property",
        "typeof",
        "type",
        "name",
        "aria-label",
        "role",
        "href",
        "src",
    ]
    cues: Dict[str, Any] = {}
    for key in cue_keys:
        if key not in node.attrs:
            continue
        if key == "class":
            cues[key] = split_classes(node.attrs[key])
        else:
            cues[key] = node.attrs[key]
    return cues


def source_kind(node: DomNode) -> str:
    if node.tag == "script" and "ld+json" in node.attrs.get("type", "").lower():
        return "embedded-jsonld"
    if node.tag in {"base", "meta", "link", "title", "script", "style"}:
        return "source-tag"
    if node.tag == "tr":
        return "table-row"
    if not evidence_text_for_node(node) and dom_cue_text(node):
        return "dom-cue"
    return "visible"


def first_block_text(blocks: List[Dict[str, Any]], html_tag: str) -> str:
    for block in blocks:
        if block.get("html_tag") == html_tag:
            return block.get("text", "")
    return ""


def make_text_from_json(obj: Any) -> str:
    values = []
    for _, value in flatten_fields(obj).items():
        if isinstance(value, str) and value:
            values.append(value)
    return "\n".join(values)


def flatten_fields(obj: Any, prefix: str = "", depth: int = 0) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if depth > 2:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "@context":
                continue
            clean_key = key.split(":")[-1]
            full_key = f"{prefix}.{clean_key}" if prefix else clean_key
            if isinstance(value, (str, int, float, bool)):
                out[full_key] = str(value)
            elif isinstance(value, dict):
                simple = first_simple_value(value)
                if simple:
                    out[full_key] = simple
                out.update(flatten_fields(value, full_key, depth + 1))
            elif isinstance(value, list):
                simple_values = [first_simple_value(v) for v in value]
                simple_values = [v for v in simple_values if v]
                if simple_values:
                    out[full_key] = " | ".join(simple_values[:5])
                for i, item in enumerate(value[:3]):
                    out.update(flatten_fields(item, f"{full_key}.{i}", depth + 1))
    return out


def first_simple_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("name", "text", "url", "price", "value", "@id"):
            if key in value and isinstance(value[key], (str, int, float, bool)):
                return str(value[key])
    return ""


def normalize_text(value: str) -> str:
    value = html.unescape(str(value)).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def supported_by_text(value: str, source_text: str) -> bool:
    norm_value = normalize_text(value)
    norm_source = normalize_text(source_text)
    if not norm_value:
        return True
    currency_support = {"gbp": "£", "usd": "$", "eur": "€"}
    if norm_value in currency_support and currency_support[norm_value] in source_text:
        return True
    if norm_value.endswith("schema org instock") and "in stock" in norm_source:
        return True
    if norm_value.endswith("schema org outofstock") and "out of stock" in norm_source:
        return True
    if len(norm_value) <= 2:
        return True
    if norm_value in norm_source:
        return True
    tokens = [t for t in norm_value.split() if len(t) > 2]
    if not tokens:
        return True
    hits = sum(1 for token in tokens if token in norm_source)
    return hits / max(1, len(tokens)) >= 0.65


def top_lines(text: str, max_lines: int = 12) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()][:max_lines]


def extract_regex_candidates(text: str) -> Dict[str, str]:
    lines = top_lines(text)
    joined = "\n".join(lines) if lines else text
    candidates: Dict[str, str] = {}
    if lines:
        candidates["name"] = lines[0][:120]
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 35]
    if sentences:
        candidates["description"] = max(sentences, key=len)[:320]
    url = re.search(r"https?://[^\s\"'<>]+", text)
    if url:
        candidates["url"] = url.group(0).rstrip(").,")
    email = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
    if email:
        candidates["email"] = email.group(0)
    phone = re.search(r"(?:\+?\d[\d .()/-]{7,}\d)", text)
    if phone:
        candidates["telephone"] = phone.group(0).strip()
    price = re.search(r"([$€£]\s?\d+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP))", text)
    if price:
        candidates["price"] = price.group(0)
    date = re.search(
        r"\b(?:20\d{2}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2})\b",
        text,
        flags=re.I,
    )
    if date:
        candidates["startDate"] = date.group(0)
    isbn = re.search(r"\b(?:ISBN(?:-1[03])?:?\s*)?97[89][-\d ]{10,}\b", text, flags=re.I)
    if isbn:
        candidates["isbn"] = isbn.group(0)
    return candidates


def baseline_extract(example: Dict[str, Any]) -> Dict[str, Any]:
    pred: Dict[str, Any] = {"@context": "https://schema.org", "@type": example["schema_type"]}
    pred.update(extract_regex_candidates(example["text"]))
    # Generic extractors often over-produce common fields without schema validation.
    if "url" in pred:
        pred["sameAs"] = pred["url"]
    if "description" not in pred and "name" in pred:
        pred["description"] = pred["name"]
    return pred


def planned_schema_rag_extract(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    gpt_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    candidate_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    repair_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    require_llm_steps: bool = True,
) -> Dict[str, Any]:
    graph = evidence_graph_for_example(
        example,
        max_blocks=(
            int(os.environ.get("SCHEMARAG_SWDE_TEMPLATE_FULL_DOM_BLOCKS", "20000"))
            if is_swde_template_example(example)
            else None
        ),
    )
    contract = plan_extraction_contract(
        example,
        schema_index,
        evidence_graph=graph,
        historical_error_patterns=example.get("historical_error_patterns", []),
        gpt_transport=gpt_transport,
        require_llm=require_llm_steps,
    )
    target_types = contract.get("target_types", [example["schema_type"]])
    schema_type = str(target_types[0]) if isinstance(target_types, list) and target_types else str(example["schema_type"])
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    candidate_allowed_paths = contract_allowed_paths(contract, schema_type, allowed)
    final_target_paths = target_slot_paths_for_example(example, schema_type, allowed)
    final_allowed_paths = final_admission_paths_for_example(example, schema_type, allowed, contract)
    target_slots, enrichment_slots = split_contract_slots_by_final_admission(contract, final_target_paths)
    constrained_candidates = generate_constrained_candidate_records(
        example,
        schema_index,
        contract,
        graph,
        candidate_transport=candidate_transport,
        require_llm=require_llm_steps,
    )
    accepted_candidates, repair_trace = verify_and_repair_candidate_records(
        constrained_candidates,
        contract,
        schema_type,
        allowed,
        candidate_allowed_paths,
        graph,
        example=example,
        repair_transport=repair_transport,
        require_llm=require_llm_steps,
    )
    pred: Dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type}
    evidence: Dict[str, str] = {}
    raw_values: Dict[str, str] = {}
    final_accepted_candidates: List[Dict[str, Any]] = []
    enrichment_candidates: List[Dict[str, Any]] = []
    final_admission_rejected: List[Dict[str, Any]] = []
    for candidate in accepted_candidates:
        path = str(candidate.get("path", "")).strip()
        if not path or path in {"@context", "@type"}:
            continue
        if not is_final_admissible_path(path, final_target_paths, allow_parent_container=False):
            admitted = dict(candidate)
            admitted["admission"] = "enrichment"
            admitted["admission_reason"] = "verified candidate retained as enrichment but excluded from final target JSON-LD"
            enrichment_candidates.append(admitted)
            final_admission_rejected.append(admitted)
            continue
        if has_jsonld_path(pred, path):
            continue
        final_accepted_candidates.append(candidate)
        value_to_set = candidate.get("value")
        raw_value = candidate.get("raw_value")
        if (
            path in {"name", "description"}
            and raw_value is not None
            and str(raw_value).strip()
            and len(str(raw_value).strip()) > len(str(value_to_set or "").strip())
            and supported_by_text(str(value_to_set), str(raw_value))
        ):
            value_to_set = str(raw_value).strip()
        set_jsonld_path(pred, path, value_to_set)
        if candidate.get("evidence_id"):
            evidence[path] = str(candidate["evidence_id"])
        if raw_value is not None:
            raw_values[path] = str(raw_value)

    # Conservative compatibility pass: if a constrained candidate abstained but the
    # deterministic slot extractor finds supported evidence, keep the old fallback.
    slot_filled = fill_missing_slots_from_property_evidence(
        example,
        contract,
        graph,
        pred,
        evidence,
        raw_values,
        allowed,
        final_target_paths,
    )
    fallback_pred = schema_rag_extract(example, schema_index)
    fallback_filled = fuse_conservative_fallback(pred, evidence, fallback_pred, graph, allowed_paths=final_target_paths)
    unsupported_pruned = prune_unsupported_jsonld_fields(pred, evidence, example.get("text", ""), raw_values)
    for path in list(raw_values):
        if not has_jsonld_path(pred, path):
            raw_values.pop(path, None)
    return {
        "jsonld": pred,
        "evidence": evidence,
        "field_provenance": evidence,
        "field_raw_values": raw_values,
        "contract": contract,
        "target_slots": target_slots,
        "enrichment_slots": enrichment_slots,
        "candidate_allowed_paths": sorted(candidate_allowed_paths),
        "final_target_paths": sorted(final_target_paths),
        "final_allowed_paths": sorted(final_allowed_paths),
        "candidates": constrained_candidates,
        "accepted_candidates": accepted_candidates,
        "verified_candidates": accepted_candidates,
        "final_accepted_candidates": final_accepted_candidates,
        "enrichment_candidates": enrichment_candidates,
        "final_admission_rejected": final_admission_rejected,
        "rejected_candidates": rejected_candidates_from_trace(repair_trace),
        "repaired_candidates": repaired_candidates_from_trace(repair_trace),
        "repairs": repair_trace,
        "slot_filled": slot_filled,
        "fallback_filled": fallback_filled,
        "unsupported_pruned": unsupported_pruned,
    }


def fill_missing_slots_from_property_evidence(
    example: Dict[str, Any],
    contract: Dict[str, Any],
    graph: Dict[str, Any],
    pred: Dict[str, Any],
    evidence: Dict[str, str],
    raw_values: Dict[str, str],
    allowed: set[str],
    allowed_paths: set[str],
) -> List[Dict[str, Any]]:
    filled: List[Dict[str, Any]] = []
    for slot in contract.get("slots", []):
        path = str(slot.get("path", "")).strip()
        if not path or path in {"@context", "@type"}:
            continue
        root = path.split(".")[0]
        if root not in allowed or path not in allowed_paths or has_jsonld_path(pred, path):
            continue
        for compact in retrieve_property_evidence(path, slot, graph, top_k=8, example=example):
            block = full_evidence_block(compact.get("evidence_id"), graph) or compact
            value = extract_value_for_slot(path, slot, block, graph)
            if not value or is_bad_slot_value(value, path, slot):
                continue
            if not candidate_value_format_ok(path, str(value), block, graph):
                continue
            template_supported = bool(compact.get("template_source"))
            if (
                not template_supported
                and not has_direct_slot_evidence(path, slot, block, str(value))
                and not has_nearby_slot_context(path, slot, block, graph)
            ):
                continue
            if not should_merge_planned_slot(path, block, slot, value):
                continue
            if not supported_by_evidence(value, block, graph):
                continue
            if not supported_by_text(value, example.get("text", "")):
                continue
            set_jsonld_path(pred, path, value)
            evidence[path] = str(block.get("evidence_id"))
            raw_value = raw_value_from_evidence_for_candidate(path, slot, str(value), block, graph)
            if raw_value:
                raw_values[path] = raw_value
            filled.append({"path": path, "value": value, "evidence_id": block.get("evidence_id"), "action": "filled_missing_slot"})
            break
    return filled


def prune_unsupported_jsonld_fields(
    pred: Dict[str, Any],
    evidence: Dict[str, str],
    source_text: str,
    raw_values: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    pruned: List[Dict[str, Any]] = []
    for path, value in list(flatten_fields(pred).items()):
        leaf = path.split(".")[-1].replace("@", "")
        if leaf in {"context", "type"}:
            continue
        if isinstance(get_jsonld_path(pred, path), dict):
            continue
        if supported_by_text(str(value), source_text):
            continue
        raw_value = (raw_values or {}).get(path)
        if raw_value and supported_by_text(str(raw_value), source_text):
            continue
        removed = delete_jsonld_path(pred, path)
        if removed:
            remove_path_metadata(evidence, path)
            if raw_values is not None:
                remove_path_metadata(raw_values, path)
            pruned.append({"path": path, "value": value, "reason": "visible-text support audit"})
    return pruned


def remove_path_metadata(metadata: Dict[str, Any], path: str) -> None:
    metadata.pop(path, None)
    prefix = f"{path}."
    for key in list(metadata):
        if key.startswith(prefix):
            metadata.pop(key, None)


def fuse_conservative_fallback(
    pred: Dict[str, Any],
    evidence: Dict[str, str],
    fallback_pred: Dict[str, Any],
    graph: Dict[str, Any],
    allowed_paths: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    filled: List[Dict[str, Any]] = []
    for path, value in flatten_fields(fallback_pred).items():
        leaf = path.split(".")[-1].replace("@", "")
        if leaf in {"context", "type"}:
            continue
        if path in {"offers", "aggregateRating"}:
            continue
        if allowed_paths is not None and path not in allowed_paths:
            continue
        if is_bad_fallback_value(path, str(value), graph):
            continue
        if get_jsonld_path(pred, path) is not None:
            continue
        block = first_supporting_block(str(value), graph)
        if block is None:
            continue
        slot = {"path": path, "evidence_query": property_evidence_terms(path)}
        if not candidate_value_format_ok(path, str(value), block, graph):
            repaired = repair_value_from_evidence(path, slot, graph)
            if not repaired:
                continue
            value, evidence_id = repaired
            block = full_evidence_block(evidence_id, graph) or block
        set_jsonld_path(pred, path, value)
        evidence[path] = str(block.get("evidence_id"))
        filled.append({"path": path, "value": value, "evidence_id": block.get("evidence_id"), "action": "filled_missing_field"})
    return filled


def is_bad_fallback_value(path: str, value: str, graph: Dict[str, Any]) -> bool:
    norm = normalize_text(value)
    if not norm:
        return True
    leaf = path.split(".")[-1].replace("@", "")
    if leaf == "name":
        first_text = ""
        for block in graph.get("blocks", []):
            first_text = str(block.get("text", "")).strip()
            if first_text:
                break
        if value.strip() == "\ufeff" or (first_text and value.strip() == first_text and len(norm.split()) <= 1):
            return True
    return False


def get_jsonld_path(obj: Dict[str, Any], dotted_path: str) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def delete_jsonld_path(obj: Dict[str, Any], dotted_path: str) -> bool:
    parts = dotted_path.split(".")
    cur: Any = obj
    parents: List[Tuple[Dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        parents.append((cur, part))
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        return False
    del cur[parts[-1]]
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and set(child.keys()) <= {"@type"}:
            del parent[key]
    return True


def field_values_equivalent(left: str, right: str) -> bool:
    if availability_equivalent(left, right):
        return True
    return values_match(normalize_text(left), normalize_text(right))


def first_supporting_block(value: str, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for block in graph.get("blocks", []):
        if supported_by_evidence(str(value), block, graph):
            return block
    return None


def rejected_candidates_from_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rejected = []
    for item in trace:
        if item.get("action") != "reject":
            continue
        candidate = dict(item.get("candidate", {}))
        candidate["reason"] = item.get("rejection_reason", item.get("repair_reason", "rejected"))
        rejected.append(candidate)
    return rejected


def repaired_candidates_from_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item["repaired"] for item in trace if item.get("action") == "repair" and isinstance(item.get("repaired"), dict)]


def candidate_record_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "value": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]},
            "raw_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "normalized_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "abstain": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "path",
            "value",
            "raw_value",
            "normalized_value",
            "evidence_id",
            "confidence",
            "abstain",
            "reason",
        ],
        "additionalProperties": False,
    }


def candidate_record_array_schema() -> Dict[str, Any]:
    return {"type": "array", "items": candidate_record_schema(), "maxItems": 24}


def candidate_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": candidate_record_array_schema(),
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def repair_record_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["accept", "replace", "move", "abstain"]},
            "path": {"type": "string"},
            "value": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]},
            "raw_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "normalized_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": ["action", "path", "value", "raw_value", "normalized_value", "evidence_id", "confidence", "reason"],
        "additionalProperties": False,
    }


def generate_constrained_candidate_records(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    contract: Dict[str, Any],
    graph: Dict[str, Any],
    candidate_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    require_llm: bool = False,
) -> List[Dict[str, Any]]:
    payload = build_candidate_generation_payload(example, schema_index, contract, graph)
    use_llm = require_llm or candidate_transport is not None or os.environ.get("SCHEMARAG_USE_GPT_CANDIDATES", "").lower() in {"1", "true", "yes"}
    records: Optional[List[Dict[str, Any]]] = None
    if use_llm:
        try:
            parse_retries = int(os.environ.get("SCHEMARAG_LLM_PARSE_RETRIES", "2"))
            for attempt in range(parse_retries + 1):
                try:
                    raw = candidate_transport(payload) if candidate_transport else call_openai_candidate_generator(payload)
                    records = normalize_candidate_records(parse_candidate_response(raw), contract, graph)
                    break
                except json.JSONDecodeError:
                    if attempt >= parse_retries:
                        raise
                    time.sleep(float(os.environ.get("SCHEMARAG_OPENAI_RETRY_BACKOFF", "2.0")) * (2**attempt))
        except Exception:
            if require_llm or os.environ.get("SCHEMARAG_CANDIDATE_STRICT", "").lower() in {"1", "true", "yes"}:
                raise
    if records is None:
        records = deterministic_constrained_candidate_records(contract, graph, example=example)
    return merge_candidate_pool_records(records, example, schema_index, contract, graph)


def merge_candidate_pool_records(
    llm_records: List[Dict[str, Any]],
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    contract: Dict[str, Any],
    graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    pooled: Dict[str, Dict[str, Any]] = {}
    for record in llm_records + source_candidate_records(contract, graph) + deterministic_fallback_candidate_records(example, schema_index, contract, graph):
        path = str(record.get("path", "")).strip()
        if not path:
            continue
        previous = pooled.get(path)
        if previous is None:
            pooled[path] = record
            continue
        if previous.get("abstain") and not record.get("abstain"):
            pooled[path] = record
            continue
        if not record.get("abstain") and safe_float(record.get("confidence"), 0.0) > safe_float(previous.get("confidence"), 0.0):
            pooled[path] = record
    ordered: List[Dict[str, Any]] = []
    for slot in contract.get("slots", []):
        if not isinstance(slot, dict):
            continue
        path = str(slot.get("path", "")).strip()
        if path and path in pooled:
            ordered.append(pooled[path])
    for path, record in pooled.items():
        if record not in ordered:
            ordered.append(record)
    return normalize_candidate_records(ordered, contract, graph)


def source_candidate_records(contract: Dict[str, Any], graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for slot in contract.get("slots", []):
        if not isinstance(slot, dict):
            continue
        path = str(slot.get("path", "")).strip()
        if not path:
            continue
        for block in graph.get("blocks", []):
            if not source_path_matches(path, block):
                continue
            value = extract_value_for_slot(path, slot, block, graph)
            if not value or is_bad_slot_value(str(value), path, slot):
                continue
            records.append(
                {
                    "path": path,
                    "value": value,
                    "raw_value": value,
                    "normalized_value": str(value),
                    "evidence_id": block.get("evidence_id"),
                    "confidence": 0.97,
                    "abstain": False,
                    "reason": "Generated from publisher source evidence",
                    "producer": "structured_source",
                }
            )
            break
    return records


def deterministic_fallback_candidate_records(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    contract: Dict[str, Any],
    graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fallback = schema_rag_extract(example, schema_index)
    allowed_paths = {
        str(slot.get("path", "")).strip()
        for slot in contract.get("slots", [])
        if isinstance(slot, dict) and slot.get("path")
    }
    records: List[Dict[str, Any]] = []
    for path, value in flatten_fields(fallback).items():
        if path in {"@context", "@type", "context", "type", "offers", "aggregateRating"}:
            continue
        nested = nested_path_for_surface_field(path) or path
        if nested not in allowed_paths:
            continue
        slot = next(
            (slot for slot in contract.get("slots", []) if isinstance(slot, dict) and str(slot.get("path", "")) == nested),
            slot_for_path(nested),
        )
        if is_bad_slot_value(str(value), nested, slot):
            continue
        block = first_supporting_block(str(value), graph)
        if not block:
            continue
        records.append(
            {
                "path": nested,
                "value": str(value),
                "raw_value": str(value),
                "normalized_value": str(value),
                "evidence_id": block.get("evidence_id"),
                "confidence": 0.72,
                "abstain": False,
                "reason": "Generated by deterministic SchemaRAG candidate producer",
                "producer": "deterministic_fallback",
            }
        )
    return records


def build_candidate_generation_payload(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    contract: Dict[str, Any],
    graph: Dict[str, Any],
) -> Dict[str, Any]:
    target_types = contract.get("target_types", [example.get("schema_type", "Thing")])
    schema_type = str(target_types[0]) if isinstance(target_types, list) and target_types else str(example.get("schema_type", "Thing"))
    return {
        "model": os.environ.get("SCHEMARAG_CANDIDATE_MODEL", PLANNER_DEFAULT_MODEL),
        "target_schema_type": schema_type,
        "candidate_record_schema": candidate_record_array_schema(),
        "candidate_response_schema": candidate_response_schema(),
        "slots": [
            {
                "path": slot.get("path"),
                "description": slot.get("description", ""),
                "value_type": slot.get("value_type", "string"),
                "required": slot.get("required", False),
                "negative_rule": slot.get("negative_rule", "Do not infer unsupported values"),
                "evidence_candidates": retrieve_property_evidence(str(slot.get("path", "")), slot, graph, example=example),
            }
            for slot in contract.get("slots", [])
            if isinstance(slot, dict) and slot.get("path")
        ],
        "schema_properties": property_descriptions(schema_type, schema_index.get(schema_type, {}).get("properties", [])),
        "instruction": (
            "Generate one constrained candidate record per slot. Return only JSON matching "
            "candidate_response_schema: an object with a candidates array. Each non-abstaining record must cite one evidence_id "
            "from that slot's evidence_candidates. If no visible evidence supports the value, "
            "return value=null, evidence_id=null, confidence=0.0, abstain=true, and reason='No visible evidence'. "
            "Do not output final JSON-LD."
        ),
    }


def call_openai_candidate_generator(payload: Dict[str, Any]) -> str:
    body = {
        "model": payload.get("model", PLANNER_DEFAULT_MODEL),
        "input": [
            {
                "role": "system",
                "content": "You generate evidence-cited schema.org slot candidates. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "candidate_records",
                "strict": True,
                "schema": candidate_response_schema(),
            }
        },
        "max_output_tokens": int(os.environ.get("SCHEMARAG_CANDIDATE_MAX_OUTPUT_TOKENS", "6000")),
    }
    return openai_response_text(body, "SCHEMARAG_CANDIDATE_TIMEOUT")


def parse_candidate_response(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        return data["candidates"]
    return data


def deterministic_constrained_candidate_records(
    contract: Dict[str, Any],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for slot in contract.get("slots", []):
        if not isinstance(slot, dict):
            continue
        path = str(slot.get("path", "")).strip()
        if not path or path in {"@context", "@type"}:
            continue
        evidence_blocks = retrieve_property_evidence(path, slot, graph, example=example)
        record: Optional[Dict[str, Any]] = None
        for compact in evidence_blocks:
            block = full_evidence_block(compact.get("evidence_id"), graph) or compact
            value = extract_value_for_slot(path, slot, block, graph)
            if not value or is_bad_slot_value(str(value), path, slot):
                continue
            if not has_direct_slot_evidence(path, slot, block):
                continue
            record = {
                "path": path,
                "value": value,
                "evidence_id": block.get("evidence_id"),
                "confidence": constrained_candidate_confidence(path, value, block, graph),
                "abstain": False,
                "reason": "Generated from slot-specific DOM evidence",
            }
            break
        if record is None:
            record = abstain_candidate_record(path, "No visible evidence")
        records.append(record)
    return normalize_candidate_records(records, contract, graph)


def normalize_candidate_records(records: Any, contract: Dict[str, Any], graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(records, list):
        return []
    contract_paths = {str(slot.get("path", "")).strip() for slot in contract.get("slots", []) if isinstance(slot, dict)}
    normalized = []
    for item in records:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        value = item.get("value")
        abstain = bool(item.get("abstain", value is None))
        evidence_id = item.get("evidence_id")
        if evidence_id is not None:
            evidence_id = str(evidence_id)
        confidence = safe_float(item.get("confidence", 0.0), 0.0)
        reason = str(item.get("reason", ""))
        if abstain:
            value = None
            evidence_id = None
            confidence = 0.0
            reason = reason or "No visible evidence"
        record = {
            "path": path,
            "value": str(value) if value is not None else None,
            "evidence_id": evidence_id,
            "confidence": max(0.0, min(1.0, confidence)),
            "abstain": abstain,
            "reason": reason,
        }
        if item.get("raw_value") is not None:
            record["raw_value"] = str(item["raw_value"])
        if item.get("normalized_value") is not None:
            record["normalized_value"] = str(item["normalized_value"])
        if path in contract_paths or nested_path_for_surface_field(path):
            normalized.append(record)
    return normalized


def abstain_candidate_record(path: str, reason: str) -> Dict[str, Any]:
    return {
        "path": path,
        "value": None,
        "evidence_id": None,
        "confidence": 0.0,
        "abstain": True,
        "reason": reason,
    }


def constrained_candidate_confidence(path: str, value: Any, block: Dict[str, Any], graph: Dict[str, Any]) -> float:
    score = property_evidence_score(path, {"path": path}, block, graph)
    if supported_by_evidence(str(value), block, graph):
        score += 4
    if block.get("table_context", {}).get("headers"):
        score += 2
    return min(0.98, 0.45 + (score / 40.0))


def verify_and_repair_candidate_records(
    candidates: List[Dict[str, Any]],
    contract: Dict[str, Any],
    schema_type: str,
    allowed: set[str],
    allowed_paths: set[str],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
    repair_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    require_llm: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    slots_by_path = {str(slot.get("path", "")).strip(): slot for slot in contract.get("slots", []) if isinstance(slot, dict)}
    accepted: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate = resolve_candidate_against_dom(candidate, slots_by_path, graph)
        ok, reason = verify_candidate_record(candidate, slots_by_path, allowed, allowed_paths, graph)
        if ok:
            accepted.append(candidate)
            trace.append({"action": "accept", "candidate": candidate, "reason": "verified"})
            continue
        if require_llm or repair_transport is not None or os.environ.get("SCHEMARAG_USE_GPT_REPAIR", "").lower() in {"1", "true", "yes"}:
            repaired = llm_repair_candidate_record(
                candidate,
                reason,
                slots_by_path,
                allowed,
                allowed_paths,
                graph,
                example=example,
                repair_transport=repair_transport,
                require_llm=require_llm,
            )
        else:
            repaired = repair_candidate_record(candidate, reason, slots_by_path, allowed, allowed_paths, graph, example=example)
        repair_ok, repair_reason = verify_candidate_record(repaired, slots_by_path, allowed, allowed_paths, graph)
        trace.append(
            {
                "action": "repair" if repair_ok else "reject",
                "rejection_reason": reason,
                "candidate": candidate,
                "repaired": repaired,
                "repair_reason": repaired.get("reason", repair_reason),
            }
        )
        if repair_ok:
            accepted.append(repaired)
    return accepted, trace


def resolve_candidate_against_dom(
    candidate: Dict[str, Any],
    slots_by_path: Dict[str, Dict[str, Any]],
    graph: Dict[str, Any],
) -> Dict[str, Any]:
    path = str(candidate.get("path", "")).strip()
    value = candidate.get("value")
    if not path or candidate.get("abstain") or value is None or not str(value).strip():
        return candidate
    slot = slots_by_path.get(path) or slots_by_path.get(nested_path_for_surface_field(path), {}) or {
        "path": path,
        "evidence_query": property_evidence_terms(path),
    }
    block = best_gxr_block_for_candidate(path, slot, str(value), graph, candidate.get("evidence_id"))
    if not block:
        return candidate
    resolved = dict(candidate)
    if str(block.get("evidence_id")) != str(candidate.get("evidence_id", "")):
        resolved["repair_action"] = "gxr_resolved_evidence"
        resolved["reason"] = f"{candidate.get('reason', 'candidate')} GXR-resolved to supporting DOM evidence"
    resolved["evidence_id"] = str(block.get("evidence_id"))
    canonical_value = canonical_value_from_evidence(path, slot, block, graph)
    if canonical_value:
        resolved["value"] = canonical_value
        resolved.setdefault("normalized_value", canonical_value)
    raw_value = raw_value_from_evidence_for_candidate(path, slot, str(resolved.get("value") or value), block, graph)
    if raw_value:
        resolved["raw_value"] = raw_value
        resolved.setdefault("normalized_value", str(value))
    return resolved


def canonical_value_from_evidence(
    path: str,
    slot: Dict[str, Any],
    block: Dict[str, Any],
    graph: Dict[str, Any],
) -> str:
    leaf = path.split(".")[-1]
    if leaf not in {"color", "depth", "height", "mpn", "sku", "width"}:
        return ""
    value = extract_value_for_slot(path, slot, block, graph)
    if not value or is_bad_slot_value(str(value), path, slot):
        return ""
    if not candidate_value_format_ok(path, str(value), block, graph):
        return ""
    if not supported_by_evidence(str(value), block, graph):
        return ""
    return str(value)


def best_gxr_block_for_candidate(
    path: str,
    slot: Dict[str, Any],
    value: str,
    graph: Dict[str, Any],
    preferred_evidence_id: Any = None,
) -> Optional[Dict[str, Any]]:
    preferred = full_evidence_block(preferred_evidence_id, graph)
    if (
        preferred
        and gxr_value_supported_by_block(value, preferred, graph)
        and gxr_value_supported_by_text(value, str(preferred.get("text", "")))
        and not is_attribute_cue_only(str(preferred.get("text", "")))
    ):
        return preferred
    scored: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for block in graph.get("blocks", []):
        score = gxr_block_score(path, slot, value, block, graph)
        if score > 0:
            scored.append((score, -int(block.get("dom_depth", 0) or 0), str(block.get("evidence_id", "")), block))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return scored[0][3]


def gxr_block_score(path: str, slot: Dict[str, Any], value: str, block: Dict[str, Any], graph: Dict[str, Any]) -> int:
    if not gxr_value_supported_by_block(value, block, graph):
        return 0
    score = 20
    text = str(block.get("text", ""))
    direct_text_support = gxr_value_supported_by_text(value, text)
    if normalize_text(value) and normalize_text(value) in normalize_text(text):
        score += 20
    if compact_digits(value) and compact_digits(value) == compact_digits(text):
        score += 14
    if has_direct_slot_evidence(path, slot, block, value):
        score += 12
    score += min(12, property_evidence_score(path, slot, block, graph))
    if is_attribute_cue_only(text):
        score -= 18 if direct_text_support else 30
    if block.get("source_kind") in {"table-row", "visible", "dom-cue"}:
        score += 2
    return score


def gxr_value_supported_by_block(value: str, block: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    raw_parts = [
        str(block.get("text", "")),
        " ".join(nearby_evidence_texts(block, graph, limit=4)),
        json.dumps(block.get("attribute_cues", {}), ensure_ascii=False),
        str(block.get("css_path", "")),
    ]
    return gxr_value_supported_by_text(value, " ".join(raw_parts))


def gxr_value_supported_by_text(value: str, text: str) -> bool:
    if supported_by_text(value, text):
        return True
    value_digits = compact_digits(value)
    text_digits = compact_digits(text)
    if value_digits and len(value_digits) >= 3 and value_digits in text_digits:
        return True
    value_compact = compact_alnum(value)
    text_compact = compact_alnum(text)
    if value_compact and len(value_compact) >= 4 and value_compact in text_compact:
        return True
    return False


def compact_digits(value: str) -> str:
    return "".join(re.findall(r"\d+", str(value)))


def compact_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def raw_value_from_evidence_for_candidate(
    path: str,
    slot: Dict[str, Any],
    value: str,
    block: Dict[str, Any],
    graph: Dict[str, Any],
) -> str:
    text = str(block.get("text", "")).strip()
    if is_attribute_cue_only(text):
        neighbor = adjacent_supported_value_text(block, graph, value)
        return neighbor[:240] if neighbor else str(value).strip()
    leaf = path.split(".")[-1]
    if leaf == "price":
        for match in re.finditer(r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?\s?(?:USD|EUR|GBP)", text, re.I):
            raw = match.group(0).strip()
            if compact_digits(raw) == compact_digits(value):
                return raw
    aliases = FIELD_ALIASES.get(leaf, []) + [leaf]
    for alias in aliases:
        match = re.search(rf"(?im)\b{re.escape(alias)}\b\s*[:\-]\s*(.+)$", text)
        if match:
            raw = match.group(1).strip()[:240]
            if gxr_value_supported_by_text(value, raw):
                return raw
    if gxr_value_supported_by_text(value, text) and len(text) <= 240:
        return text
    neighbor = adjacent_value_text(block, graph, path, slot)
    if neighbor and gxr_value_supported_by_text(value, neighbor):
        return neighbor[:240]
    return ""


def adjacent_supported_value_text(block: Dict[str, Any], graph: Dict[str, Any], value: str) -> str:
    evidence_id = block.get("evidence_id")
    by_id = {b.get("evidence_id"): b for b in graph.get("blocks", [])}
    neighbor_ids = sequential_neighbor_ids(evidence_id, graph, before=4, after=4)
    for edge in graph.get("edges", []):
        if edge.get("source") == evidence_id:
            neighbor_ids.append(edge.get("target"))
        elif edge.get("target") == evidence_id:
            neighbor_ids.append(edge.get("source"))
    seen = set()
    for neighbor_id in neighbor_ids:
        if neighbor_id in seen:
            continue
        seen.add(neighbor_id)
        neighbor = by_id.get(neighbor_id)
        if not neighbor:
            continue
        text = str(neighbor.get("text", "")).strip()
        if text and not is_attribute_cue_only(text) and gxr_value_supported_by_text(value, text):
            return text
    return ""


def verify_candidate_record(
    candidate: Dict[str, Any],
    slots_by_path: Dict[str, Dict[str, Any]],
    allowed: set[str],
    allowed_paths: set[str],
    graph: Dict[str, Any],
) -> Tuple[bool, str]:
    path = str(candidate.get("path", "")).strip()
    if not path:
        return False, "missing path"
    if candidate.get("abstain"):
        return False, "candidate abstained"
    value = candidate.get("value")
    if value is None or not str(value).strip():
        return False, "missing value"
    root = path.split(".")[0]
    if root not in allowed:
        if nested_path_for_surface_field(path):
            return False, "wrong nesting"
        return False, "schema-invalid path"
    if path not in allowed_paths and path not in slots_by_path:
        return False, "path outside extraction contract"
    evidence_id = candidate.get("evidence_id")
    block = full_evidence_block(evidence_id, graph)
    if not block:
        return False, "missing evidence"
    slot = slots_by_path.get(path, {"path": path, "evidence_query": property_evidence_terms(path)})
    if is_bad_slot_value(str(value), path, slot):
        return False, "bad slot value"
    if not candidate_value_format_ok(path, str(value), block, graph):
        return False, "bad candidate format"
    if not has_direct_slot_evidence(path, slot, block, str(value)) and not has_nearby_slot_context(path, slot, block, graph):
        return False, "weak slot evidence"
    if not supported_by_evidence(str(value), block, graph):
        return False, "evidence mismatch"
    if not should_merge_planned_slot(path, block, slot, str(value)):
        return False, "unsafe generic slot"
    return True, "ok"


def candidate_value_format_ok(path: str, value: str, block: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    leaf = path.split(".")[-1]
    raw_context = " ".join(
        [
            str(block.get("text", "")),
            " ".join(nearby_evidence_texts(block, graph, limit=4)),
            json.dumps(block.get("attribute_cues", {}), ensure_ascii=False),
            str(block.get("css_path", "")),
        ]
    )
    stripped = value.strip()
    if leaf == "sku":
        if len(stripped) > 80:
            return False
        upc = upc_value_from_text(raw_context)
        if upc and compact_digits(stripped) != upc:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/#-]{3,79}", stripped))
    if leaf == "mpn":
        if len(stripped) > 120:
            return False
        upc = upc_value_from_text(raw_context)
        if upc and compact_digits(stripped) == upc:
            return False
        return bool(re.search(r"[A-Za-z0-9]", stripped))
    if leaf in {"width", "height", "depth"}:
        return 0 < len(stripped) <= 40 and bool(first_dimension_value(stripped))
    if leaf == "category":
        return 0 < len(stripped) <= 80 and not re.search(r"[.!?].{20,}", stripped)
    if leaf == "material":
        return 0 < len(stripped) <= 120 and not re.search(r"[.!?].{20,}", stripped)
    if leaf == "size":
        return 0 < len(stripped) <= 80 and not re.search(r"[.!?].{20,}", stripped)
    if leaf == "availability":
        return bool(availability_uri_from_text(stripped))
    if leaf == "price":
        has_price_cue = bool(re.search(r"(?i)\b(?:price|msrp|sale|our price|market price)\b|[$€£]|\b(?:USD|EUR|GBP)\b", raw_context))
        return has_price_cue and bool(re.search(r"[$€£]\s?\d+|\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)", stripped, re.I))
    if leaf == "priceCurrency":
        return bool(currency_code_from_text(stripped))
    if leaf == "color":
        return 0 < len(stripped) <= 80 and not re.search(
            r"(?i)\b(?:UPC|SKU|MPN|MODEL|BRAND|MANUFACTURER|PRICE|AVAILABILITY)\b\s*:",
            stripped,
        )
    if leaf == "ratingValue":
        return bool(star_rating_value(stripped) or re.search(r"\b\d+(?:\.\d+)?\b", stripped))
    if leaf == "reviewCount":
        return bool(re.search(r"\b\d+\b", stripped))
    return True


def llm_repair_candidate_record(
    candidate: Dict[str, Any],
    reason: str,
    slots_by_path: Dict[str, Dict[str, Any]],
    allowed: set[str],
    allowed_paths: set[str],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
    repair_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    require_llm: bool = False,
) -> Dict[str, Any]:
    payload = build_repair_payload(candidate, reason, slots_by_path, allowed, allowed_paths, graph, example=example)
    try:
        parse_retries = int(os.environ.get("SCHEMARAG_LLM_PARSE_RETRIES", "2"))
        for attempt in range(parse_retries + 1):
            try:
                raw = repair_transport(payload) if repair_transport else call_openai_repair(payload)
                repaired = normalize_repair_response(parse_repair_response(raw), candidate)
                return repaired
            except json.JSONDecodeError:
                if attempt >= parse_retries:
                    raise
                time.sleep(float(os.environ.get("SCHEMARAG_OPENAI_RETRY_BACKOFF", "2.0")) * (2**attempt))
    except Exception:
        if require_llm or os.environ.get("SCHEMARAG_REPAIR_STRICT", "").lower() in {"1", "true", "yes"}:
            raise
    return repair_candidate_record(candidate, reason, slots_by_path, allowed, allowed_paths, graph, example=example)


def build_repair_payload(
    candidate: Dict[str, Any],
    reason: str,
    slots_by_path: Dict[str, Dict[str, Any]],
    allowed: set[str],
    allowed_paths: set[str],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = str(candidate.get("path", "")).strip()
    nested_path = nested_path_for_surface_field(path)
    repair_path = nested_path or path
    slot = slots_by_path.get(repair_path) or slots_by_path.get(path) or {
        "path": repair_path,
        "evidence_query": property_evidence_terms(repair_path),
        "description": f"Repair {repair_path} only from evidence",
    }
    evidence_ids = []
    if candidate.get("evidence_id"):
        evidence_ids.append(str(candidate["evidence_id"]))
    for compact in retrieve_property_evidence(repair_path, slot, graph, top_k=6, example=example):
        eid = str(compact.get("evidence_id", ""))
        if eid and eid not in evidence_ids:
            evidence_ids.append(eid)
    evidence = [
        compact_evidence_block(block, graph)
        for block in graph.get("blocks", [])
        if str(block.get("evidence_id")) in evidence_ids
    ]
    return {
        "model": os.environ.get("SCHEMARAG_REPAIR_MODEL", PLANNER_DEFAULT_MODEL),
        "rejected_candidate": candidate,
        "rejection_reason": reason,
        "target_path": repair_path,
        "slot_contract": slot,
        "allowed_top_level_properties": sorted(allowed),
        "allowed_paths": sorted(allowed_paths),
        "retrieved_evidence": evidence,
        "repair_record_schema": repair_record_schema(),
        "instruction": (
            "Repair exactly one rejected schema.org candidate. Return only JSON matching "
            "repair_record_schema. If the value is supported but nested incorrectly, move it "
            "to target_path. If the cited evidence contradicts the value, replace the value "
            "from evidence and cite that evidence_id. If no evidence supports a valid repair, "
            "return action='abstain', value=null, evidence_id=null, confidence=0.0. Do not invent values."
        ),
    }


def call_openai_repair(payload: Dict[str, Any]) -> str:
    body = {
        "model": payload.get("model", PLANNER_DEFAULT_MODEL),
        "input": [
            {
                "role": "system",
                "content": "You repair rejected schema.org slot candidates using verifier feedback. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "repair_record",
                "strict": True,
                "schema": repair_record_schema(),
            }
        },
        "max_output_tokens": int(os.environ.get("SCHEMARAG_REPAIR_MAX_OUTPUT_TOKENS", "2000")),
    }
    return openai_response_text(body, "SCHEMARAG_REPAIR_TIMEOUT")


def parse_repair_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    if isinstance(data, dict) and isinstance(data.get("repair"), dict):
        return data["repair"]
    return data if isinstance(data, dict) else {}


def normalize_repair_response(record: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        record = {}
    action = str(record.get("action", "abstain")).strip() or "abstain"
    path = str(record.get("path") or candidate.get("path") or "").strip()
    value = record.get("value")
    abstain = action == "abstain" or value is None
    evidence_id = record.get("evidence_id")
    if evidence_id is not None:
        evidence_id = str(evidence_id)
    confidence = safe_float(record.get("confidence", 0.0), 0.0)
    if abstain:
        return {
            "path": path,
            "value": None,
            "evidence_id": None,
            "confidence": 0.0,
            "abstain": True,
            "repair_action": "abstained_due_to_no_evidence",
            "reason": str(record.get("reason") or "LLM repair abstained"),
        }
    repair_action = {
        "move": "moved_to_nested_path",
        "replace": "corrected_value_from_evidence",
        "accept": "accepted_after_feedback",
    }.get(action, action)
    return {
        "path": path,
        "value": str(value),
        "evidence_id": evidence_id,
        "confidence": max(0.0, min(1.0, confidence)),
        "abstain": False,
        "repair_action": repair_action,
        "reason": str(record.get("reason") or "LLM verifier-guided repair"),
        **({"raw_value": str(record["raw_value"])} if record.get("raw_value") is not None else {}),
        **({"normalized_value": str(record["normalized_value"])} if record.get("normalized_value") is not None else {}),
    }


def repair_candidate_record(
    candidate: Dict[str, Any],
    reason: str,
    slots_by_path: Dict[str, Dict[str, Any]],
    allowed: set[str],
    allowed_paths: set[str],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = str(candidate.get("path", "")).strip()
    repaired = dict(candidate)
    moved = False
    if reason == "wrong nesting":
        nested = nested_path_for_surface_field(path)
        if nested:
            repaired["path"] = nested
            repaired["reason"] = f"Moved {path} to {nested}"
            path = nested
            moved = True
    if reason in {"evidence mismatch", "missing evidence", "bad slot value", "bad candidate format", "weak slot evidence", "candidate abstained", "missing value", "wrong nesting"}:
        slot = slots_by_path.get(path, {"path": path, "evidence_query": property_evidence_terms(path)})
        replacement = repair_value_from_candidate_evidence(path, slot, candidate.get("evidence_id"), graph)
        if not replacement:
            replacement = repair_value_from_evidence(path, slot, graph, example=example)
        if replacement:
            value, evidence_id = replacement
            block = full_evidence_block(evidence_id, graph)
            raw_value = raw_value_from_evidence_for_candidate(path, slot, str(value), block, graph) if block else ""
            repair_action = "moved_to_nested_path" if moved else "corrected_value_from_evidence"
            if moved and str(candidate.get("value", "")) != str(value):
                repair_action = "moved_to_nested_path_and_corrected_value"
            repaired.update(
                {
                    "path": path,
                    "value": value,
                    "evidence_id": evidence_id,
                    "confidence": max(0.5, safe_float(repaired.get("confidence", 0.0), 0.0)),
                    "abstain": False,
                    "repair_action": repair_action,
                    "reason": f"Verifier-guided repair after {reason}",
                }
            )
            if raw_value:
                repaired["raw_value"] = raw_value
                repaired["normalized_value"] = str(value)
            return repaired
    repaired.update(abstain_candidate_record(path or str(candidate.get("path", "")), f"Rejected after verifier feedback: {reason}"))
    repaired["repair_action"] = "abstained_due_to_no_evidence"
    return repaired


def repair_value_from_candidate_evidence(
    path: str,
    slot: Dict[str, Any],
    evidence_id: Any,
    graph: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    block = full_evidence_block(evidence_id, graph)
    if not block:
        return None
    value = extract_value_for_slot(path, slot, block, graph)
    if not value or is_bad_slot_value(str(value), path, slot):
        return None
    if not has_direct_slot_evidence(path, slot, block, str(value)) and not has_nearby_slot_context(path, slot, block, graph):
        return None
    if supported_by_evidence(str(value), block, graph) and should_merge_planned_slot(path, block, slot, str(value)):
        return str(value), str(block.get("evidence_id"))
    return None


def repair_value_from_evidence(
    path: str,
    slot: Dict[str, Any],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str]]:
    for compact in retrieve_property_evidence(path, slot, graph, top_k=8, example=example):
        block = full_evidence_block(compact.get("evidence_id"), graph) or compact
        value = extract_value_for_slot(path, slot, block, graph)
        if not value or is_bad_slot_value(str(value), path, slot):
            continue
        if not has_direct_slot_evidence(path, slot, block, str(value)) and not has_nearby_slot_context(path, slot, block, graph):
            continue
        if supported_by_evidence(str(value), block, graph) and should_merge_planned_slot(path, block, slot, str(value)):
            return str(value), str(block.get("evidence_id"))
    return None


def source_path_matches(path: str, block: Dict[str, Any]) -> bool:
    source_path = str(block.get("attribute_cues", {}).get("source_path", "")).strip()
    if not source_path:
        return False
    return source_path == path or source_path.split(".")[-1].lower() == path.split(".")[-1].lower()


def strict_identity_property_evidence(path: str, block: Dict[str, Any]) -> bool:
    leaf = path.split(".")[-1].lower()
    text = str(block.get("text", ""))
    headers = " ".join(block.get("table_context", {}).get("headers", []))
    attrs = json.dumps(block.get("attribute_cues", {}), ensure_ascii=False)
    css_path = str(block.get("css_path", ""))
    raw = " ".join([text, headers, attrs, css_path])
    labels = {
        "manufacturer": r"(?:manufacturer|mfr|brand|make|maker)",
        "model": r"(?:model|modelname|model_vch)",
        "brand": r"(?:brand|maker|make)",
        "mpn": r"(?:mpn|manufacturer stock number|manufacturer part number|part number|model number|mfr part|item #|alternate item #)",
    }.get(leaf, re.escape(leaf))
    if leaf == "mpn" and block.get("html_tag") in {"title", "h1"} and re.search(r"\s-\s*[A-Za-z]{2,}[A-Za-z0-9._-]{3,}\b", raw):
        return True
    if leaf in {"brand", "manufacturer"} and block.get("html_tag") in {"title", "h1"} and brand_value_from_title(raw):
        return True
    return bool(
        re.search(rf"(?i)\b{labels}\b\s*[:\-]", raw)
        or re.search(rf"(?i)(?:itemprop|property)[\"':\s=]+{re.escape(leaf)}\b", raw)
    )


def has_direct_slot_evidence(path: str, slot: Dict[str, Any], block: Dict[str, Any], value: str = "") -> bool:
    if block.get("source_kind") in {"structured-jsonld", "source-attribute"} and source_path_matches(path, block):
        return True
    if path in STRICT_IDENTITY_EVIDENCE_PATHS:
        return strict_identity_property_evidence(path, block)
    text = str(block.get("text", ""))
    headers = " ".join(block.get("table_context", {}).get("headers", []))
    attrs = json.dumps(block.get("attribute_cues", {}), ensure_ascii=False)
    css_path = str(block.get("css_path", ""))
    raw = " ".join([text, headers, attrs, css_path])
    norm_raw = normalize_text(raw)
    leaf = path.split(".")[-1]
    if leaf in {"name", "title"}:
        if block.get("html_tag") in {"title", "h1"}:
            return True
        if re.search(r"(?im)\b(?:name|title|job title)\b\s*[:\-]", text):
            return True
        if "itemprop" in block.get("attribute_cues", {}) and "name" in normalize_text(str(block["attribute_cues"]["itemprop"])):
            return True
    if leaf == "price":
        return bool(re.search(r"[$€£]\s?\d+|\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)", raw, re.I))
    if leaf == "priceCurrency":
        return bool(re.search(r"[$€£]|\b(?:USD|EUR|GBP)\b", raw, re.I))
    if leaf in {"telephone", "phone"}:
        return bool(re.search(r"(?:\+?\d[\d .()/-]{7,}\d)", raw))
    if leaf == "url":
        return bool(re.search(r"https?://|www\.", raw, re.I))
    if leaf == "ratingValue":
        return bool(re.search(r"\b(star|rating|stars?)\b", raw, re.I))
    if leaf == "reviewCount":
        return bool(re.search(r"\b(review count|number of reviews|reviews?)\b", raw, re.I))
    if leaf in {"sku", "mpn", "isbn"}:
        if re.search(r"\b(SKU|UPC|ISBN|MPN)\b", raw, re.I) or re.search(r"\b(?:97[89][- ]?)?\d[\d -]{8,}\d\b", value or text):
            return True
        if leaf == "sku" and block.get("html_tag") in {"title", "h1"} and re.search(r"\s-\s*[A-Za-z0-9][A-Za-z0-9._-]{4,}\b", raw):
            return True
        return False
    if value and is_self_describing_slot_value(leaf, value):
        return gxr_value_supported_by_block(value, block, {"blocks": [block], "edges": []}) or gxr_value_supported_by_text(value, raw)
    for term in property_evidence_terms(path, slot):
        clean = str(term).strip()
        norm_term = normalize_text(clean)
        if not norm_term:
            continue
        if clean in {"$", "£", "€"} and clean in raw:
            return True
        if norm_term in norm_raw:
            return True
    return False


def has_nearby_slot_context(path: str, slot: Dict[str, Any], block: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    if is_navigation_label(str(block.get("text", ""))):
        return False
    return property_evidence_score(path, slot, block, graph) >= 3


def is_self_describing_slot_value(leaf: str, value: str) -> bool:
    raw = str(value)
    if leaf in {"datePublished", "datePosted", "startDate", "endDate"}:
        return bool(re.search(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", raw, re.I))
    if leaf == "vehicleEngine":
        return bool(re.search(r"\b(?:V-?\d|I-?\d|\d\.\dL|engine|hp|horsepower|cylinder|turbo)\b", raw, re.I))
    if leaf == "fuelEfficiency":
        return bool(re.search(r"\b(?:mpg|city|hwy|highway|fuel)\b|\b\d+\s*[|/]\s*\d+\b", raw, re.I))
    if leaf == "height":
        return bool(re.search(r"\b\d+\s*(?:ft|feet|'|-)\s*\d*", raw, re.I))
    if leaf == "weight":
        return bool(re.search(r"\b\d{2,3}\s*(?:lb|lbs|pounds|kg)?\b", raw, re.I))
    if leaf == "address":
        return bool(re.search(r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|blvd|suite|city|state|zip)\b|\d{5}", raw, re.I))
    return False


def nested_path_for_surface_field(path: str) -> str:
    leaf = path.split(".")[-1]
    mapping = {
        "Product.price": "offers.price",
        "price": "offers.price",
        "currency": "offers.priceCurrency",
        "priceCurrency": "offers.priceCurrency",
        "availability": "offers.availability",
        "rating": "aggregateRating.ratingValue",
        "ratingValue": "aggregateRating.ratingValue",
        "reviewCount": "aggregateRating.reviewCount",
    }
    return mapping.get(path, mapping.get(leaf, ""))


def full_evidence_block(evidence_id: Any, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not evidence_id:
        return None
    for block in graph.get("blocks", []):
        if block.get("evidence_id") == evidence_id:
            return block
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evidence_graph_for_example(example: Dict[str, Any], max_blocks: Optional[int] = None) -> Dict[str, Any]:
    if isinstance(example.get("evidence_graph"), dict):
        return example["evidence_graph"]
    html_text = example.get("html", "")
    raw_path = example.get("raw_html_path", "")
    if not html_text and raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        max_html_bytes = int(
            os.environ.get(
                "SCHEMARAG_SWDE_TEMPLATE_MAX_HTML_BYTES" if max_blocks else "DOM_GRAPH_MAX_HTML_BYTES",
                "5000000" if max_blocks else "500000",
            )
        )
        if path.exists() and path.stat().st_size <= max_html_bytes:
            html_text = read_text(path)
    if html_text:
        return build_dom_evidence_graph(html_text, max_blocks=max_blocks or 160)
    return graph_from_plain_text(example.get("text", ""))


_SWDE_TEMPLATE_INDEX_CACHE: Dict[str, Tuple[float, Dict[Tuple[str, str, str], List[Dict[str, Any]]]]] = {}


def swde_template_index_enabled() -> bool:
    return os.environ.get("SCHEMARAG_SWDE_TEMPLATE_ENABLED", "1").lower() not in {"0", "false", "no"}


def swde_template_index_path() -> Path:
    raw = os.environ.get(
        "SCHEMARAG_SWDE_TEMPLATE_INDEX",
        "results/swde_official_train_template_probe_cap200/compiled_templates.json",
    )
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


def load_swde_template_index() -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    if not swde_template_index_enabled():
        return {}
    path = swde_template_index_path()
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cached = _SWDE_TEMPLATE_INDEX_CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    compiled: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for key, value in raw.items():
        parts = tuple(str(key).split("|"))
        if len(parts) != 3 or not isinstance(value, list):
            continue
        compiled[parts] = [dict(item) for item in value if isinstance(item, dict)]
    _SWDE_TEMPLATE_INDEX_CACHE[str(path)] = (mtime, compiled)
    return compiled


def is_swde_template_example(example: Dict[str, Any]) -> bool:
    return is_swde_example(example) and bool(load_swde_template_index())


def swde_template_feature_score(block: Dict[str, Any], template: Dict[str, Any]) -> int:
    kind, value = template.get("feature", (None, None))
    if isinstance(value, list):
        value = tuple(value)
    if kind == "xpath":
        base = 125 if str(block.get("xpath", "")) == value else 0
    elif kind == "xps5":
        base = 95 if xpath_suffix(block.get("xpath", ""), 5) == value else 0
    elif kind == "xps4":
        base = 75 if xpath_suffix(block.get("xpath", ""), 4) == value else 0
    elif kind == "tag5":
        base = 45 if tag_suffix(block.get("xpath", ""), 5) == value else 0
    elif kind == "tag4":
        base = 35 if tag_suffix(block.get("xpath", ""), 4) == value else 0
    elif kind == "css4":
        base = 100 if css_suffix(block.get("css_path", ""), 4) == value else 0
    elif kind == "css3":
        base = 80 if css_suffix(block.get("css_path", ""), 3) == value else 0
    elif kind == "header":
        base = 75 if value in table_header_signature(block) else 0
    elif kind == "attr":
        base = 70 if value in attribute_cues_signature(block) else 0
    else:
        base = 0
    if not base:
        return 0
    return (
        base
        + min(40, int(template.get("count", 0)) * 2)
        + min(25, int(float(template.get("confidence", 0.0)) * 25))
        - min(30, len(normalize_text(block.get("text", "")).split()) // 10)
    )


def attribute_cues_signature(block: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    out = []
    for key, value in sorted((block.get("attribute_cues", {}) or {}).items()):
        normalized = normalize_text(value)
        if normalized:
            out.append((key, normalized[:80]))
    return tuple(out)


def table_header_signature(block: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(
        normalize_text(header)
        for header in block.get("table_context", {}).get("headers", [])
        if normalize_text(header)
    )


def xpath_suffix(xpath: Any, depth: int = 5) -> str:
    parts = [part for part in str(xpath).split("/") if part]
    return "/".join(parts[-depth:])


def tag_suffix(xpath: Any, depth: int = 5) -> str:
    parts = [re.sub(r"\[\d+\]", "", part) for part in str(xpath).split("/") if part]
    return "/".join(parts[-depth:])


def css_suffix(css_path: Any, depth: int = 4) -> str:
    parts = [part.strip() for part in str(css_path).split(">") if part.strip()]
    return " > ".join(parts[-depth:])


def template_indexed_evidence_blocks(
    example: Dict[str, Any],
    path: str,
    graph: Dict[str, Any],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not is_swde_template_example(example):
        return []
    templates = load_swde_template_index()
    if not templates:
        return []
    top_k = top_k or int(os.environ.get("SCHEMARAG_TEMPLATE_EVIDENCE_K", "8"))
    scored: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for key in (
        (str(example.get("vertical", "")), str(example.get("site", "")), path),
        (str(example.get("vertical", "")), str(example.get("site", "")), path.split(".")[-1].replace("@", "")),
    ):
        for template in templates.get(key, []):
            for block in graph.get("blocks", []):
                score = swde_template_feature_score(block, template)
                if score and score > scored.get(str(block.get("evidence_id", "")), (0, block))[0]:
                    scored[str(block.get("evidence_id", ""))] = (score, block)
    blocks = [block for _, block in sorted(scored.values(), key=lambda item: -item[0])]
    compact_blocks = []
    for block in blocks[:top_k]:
        compact = compact_evidence_block(block, graph)
        compact["template_source"] = True
        compact_blocks.append(compact)
    return compact_blocks


def graph_from_plain_text(text: str) -> Dict[str, Any]:
    blocks = []
    lines = top_lines(text, 80)
    for idx, line in enumerate(lines, 1):
        blocks.append(
            {
                "evidence_id": f"ev_{idx:04d}",
                "text": line,
                "html_tag": "text",
                "xpath": f"/text[{idx}]",
                "css_path": f"text:nth-of-type({idx})",
                "dom_depth": 0,
                "parent_id": None,
                "child_ids": [],
                "sibling_ids": [],
                "previous_sibling_id": f"ev_{idx-1:04d}" if idx > 1 else None,
                "next_sibling_id": f"ev_{idx+1:04d}" if idx < len(lines) else None,
                "relations": {"parent": None, "children": [], "siblings": []},
                "table_context": {"in_table": False, "headers": [], "row_header": "", "table_xpath": ""},
                "attribute_cues": {},
                "source_kind": "visible",
            }
        )
    edges = [
        {"source": blocks[i]["evidence_id"], "target": blocks[i + 1]["evidence_id"], "relation": "next_sibling"}
        for i in range(len(blocks) - 1)
    ]
    return {
        "blocks": blocks,
        "edges": edges,
        "summary": {
            "block_count": len(blocks),
            "tags": ["text"],
            "source_kinds": ["visible"],
            "page_title": blocks[0]["text"] if blocks else "",
        },
    }


def plan_extraction_contract(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    evidence_graph: Optional[Dict[str, Any]] = None,
    historical_error_patterns: Optional[List[str]] = None,
    gpt_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    require_llm: bool = False,
) -> Dict[str, Any]:
    graph = evidence_graph or evidence_graph_for_example(example)
    payload = build_planner_payload(example, schema_index, graph, historical_error_patterns or [])
    use_gpt = require_llm or gpt_transport is not None or os.environ.get("SCHEMARAG_USE_GPT_PLANNER", "").lower() in {"1", "true", "yes"}
    if use_gpt:
        try:
            parse_retries = int(os.environ.get("SCHEMARAG_LLM_PARSE_RETRIES", "2"))
            for attempt in range(parse_retries + 1):
                try:
                    raw = gpt_transport(payload) if gpt_transport else call_openai_planner(payload)
                    contract = normalize_extraction_contract(parse_planner_response(raw), example, schema_index)
                    return merge_default_contract_slots(contract, example, schema_index)
                except json.JSONDecodeError:
                    if attempt >= parse_retries:
                        raise
                    time.sleep(float(os.environ.get("SCHEMARAG_OPENAI_RETRY_BACKOFF", "2.0")) * (2**attempt))
        except Exception as exc:
            if require_llm or os.environ.get("SCHEMARAG_PLANNER_STRICT", "").lower() in {"1", "true", "yes"}:
                raise
            payload["planner_error"] = f"{type(exc).__name__}: {exc}"
    return merge_default_contract_slots(fallback_extraction_contract(example, schema_index), example, schema_index)


def build_planner_payload(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    evidence_graph: Dict[str, Any],
    historical_error_patterns: List[str],
) -> Dict[str, Any]:
    schema_type = example.get("schema_type", "Thing")
    properties = schema_index.get(schema_type, {}).get("properties", [])
    page_blocks = select_page_summary_blocks(
        evidence_graph,
        int(os.environ.get("SCHEMARAG_PLANNER_PAGE_BLOCKS", "3")),
    )
    return {
        "model": os.environ.get("SCHEMARAG_PLANNER_MODEL", PLANNER_DEFAULT_MODEL),
        "target_schema_type_candidates": [schema_type],
        "schema_properties": property_descriptions(schema_type, properties),
        "page_title": evidence_graph.get("summary", {}).get("page_title", "") or first_text_line(example.get("text", "")),
        "dom_summary": evidence_graph.get("summary", {}),
        "top_dom_evidence_blocks": page_blocks,
        "property_evidence_by_slot": property_evidence_by_slot(schema_type, schema_index, evidence_graph, example=example),
        "historical_error_patterns": historical_error_patterns,
        "instruction": (
            "Return an extraction contract JSON object with target_types, slots, json_schema, "
            "nesting_rules, and abstention_rules. Use property_evidence_by_slot as the primary "
            "grounding source: each slot has its own retrieved DOM evidence blocks and nearby "
            "labels. Do not infer values from unrelated page-summary blocks. Slots must cite "
            "evidence_query terms and include negative rules forbidding unsupported inference. "
            "Keep the contract compact: return at most 6 slots, use short descriptions, and make "
            "json_schema minimal, e.g. {'type':'object'} plus only necessary properties."
        ),
    }


def select_page_summary_blocks(graph: Dict[str, Any], max_blocks: int) -> List[Dict[str, Any]]:
    selected = []
    seen = set()
    for block in graph.get("blocks", []):
        tag = block.get("html_tag", "")
        if tag in {"title", "h1"} or block.get("source_kind") == "embedded-jsonld":
            selected.append(compact_evidence_block(block, graph))
            seen.add(block.get("evidence_id"))
            if len(selected) >= max_blocks:
                return selected
    for block in graph.get("blocks", []):
        evidence_id = block.get("evidence_id")
        if evidence_id in seen:
            continue
        selected.append(compact_evidence_block(block, graph))
        if len(selected) >= max_blocks:
            break
    return selected


def property_evidence_by_slot(
    schema_type: str,
    schema_index: Dict[str, Any],
    graph: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    paths = default_contract_paths(schema_type, allowed)
    out = []
    for path in paths:
        hint = PLANNER_SLOT_HINTS.get(path) or PLANNER_SLOT_HINTS.get(path.split(".")[-1], {})
        slot = {
            "path": path,
            "description": hint.get("description", f"Extract {path} only when shown in page evidence"),
            "evidence_query": hint.get("evidence_query", property_evidence_terms(path)),
        }
        out.append(
            {
                "path": path,
                "description": slot["description"],
                "evidence_query": slot["evidence_query"],
                "evidence_blocks": retrieve_property_evidence(path, slot, graph, example=example),
            }
        )
    return out


def retrieve_property_evidence(
    path: str,
    slot: Dict[str, Any],
    graph: Dict[str, Any],
    top_k: Optional[int] = None,
    example: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    top_k = top_k or int(os.environ.get("SCHEMARAG_PROPERTY_EVIDENCE_K", "8"))
    template_blocks = template_indexed_evidence_blocks(example, path, graph, top_k=top_k) if example else []
    seen = {block.get("evidence_id") for block in template_blocks}
    scored = []
    for block in graph.get("blocks", []):
        if block.get("evidence_id") in seen:
            continue
        score = property_evidence_score(path, slot, block, graph)
        if score > 0:
            scored.append((score, block.get("dom_depth", 0), block))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].get("evidence_id", "")))
    combined = list(template_blocks)
    for _, _, block in scored:
        if len(combined) >= top_k:
            break
        combined.append(compact_evidence_block(block, graph))
    if path.split(".")[-1] in PRODUCT_CONTEXT_FALLBACK_PATHS and len(combined) < min(top_k, 4):
        for block in product_context_fallback_blocks(graph):
            evidence_id = block.get("evidence_id")
            if evidence_id in seen or any(item.get("evidence_id") == evidence_id for item in combined):
                continue
            combined.append(compact_evidence_block(block, graph))
            if len(combined) >= min(top_k, 4):
                break
    return combined


def product_context_fallback_blocks(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    preferred: List[Dict[str, Any]] = []
    for block in graph.get("blocks", []):
        tag = str(block.get("html_tag", "")).lower()
        css_path = str(block.get("css_path", "")).lower()
        if tag in {"title", "h1"}:
            preferred.append(block)
            continue
        if tag == "meta" and "description" in css_path:
            preferred.append(block)
            continue
        if "description" in css_path or "product" in css_path:
            preferred.append(block)
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for block in preferred:
        evidence_id = str(block.get("evidence_id", ""))
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        deduped.append(block)
    return deduped[:6]


def property_evidence_score(
    path: str,
    slot: Dict[str, Any],
    block: Dict[str, Any],
    graph: Dict[str, Any],
) -> int:
    terms = property_evidence_terms(path, slot)
    nearby = " ".join(nearby_evidence_texts(block, graph, limit=6))
    text = block.get("text", "")
    headers = " ".join(block.get("table_context", {}).get("headers", []))
    attrs = json.dumps(block.get("attribute_cues", {}), ensure_ascii=False)
    css_path = block.get("css_path", "")
    raw_haystack = " ".join([text, headers, attrs, css_path, nearby])
    norm_haystack = normalize_text(raw_haystack)
    norm_text = normalize_text(text)
    norm_nearby = normalize_text(nearby)
    norm_label_haystack = normalize_text(" ".join([headers, attrs, css_path]))
    score = 0
    if is_navigation_label(text) or re.search(r"(?i)^\s*(?:for publishers|browse categories|company info|privacy policy|career opportunities|store finder|contact us)\s*$", text):
        score -= 14
    for term in terms:
        raw_term = str(term)
        norm_term = normalize_text(raw_term)
        if raw_term in {"$", "£", "€"} and raw_term in raw_haystack:
            score += 4
            continue
        if not norm_term:
            continue
        if norm_term in norm_label_haystack:
            score += 5
        if norm_term in norm_text:
            score += 4
        elif norm_term in norm_nearby and not is_any_field_label(text):
            score += 6
        elif norm_term in norm_haystack:
            score += 2
    leaf = path.split(".")[-1]
    if leaf in {"name", "title"}:
        if block.get("html_tag") in {"title", "h1"}:
            score += 12
        if "itemprop" in block.get("attribute_cues", {}) and "name" in normalize_text(str(block["attribute_cues"]["itemprop"])):
            score += 8
        if "username" in norm_haystack:
            score -= 12
    if leaf == "price" and re.search(r"[$€£]\s?\d+|\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)", raw_haystack, re.I):
        score += 8
        if re.search(r"\b(?:MSRP|starting MSRP|market price|sale price|our price|price)\b", raw_haystack, re.I):
            score += 6
    if leaf == "model":
        if block.get("html_tag") in {"title", "h1"}:
            score += 10
        if re.search(r"\bmodel(?:name|_vch)?\b", raw_haystack, re.I):
            score += 8
        if re.search(r"\b(?:19|20)\d{2}\b", text) and 2 <= len(norm_text.split()) <= 12:
            score += 5
    if leaf in {"datePublished", "datePosted"}:
        if re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", raw_haystack, re.I):
            score += 10
        if re.search(r"\b(?:published|publication|pub\.?\s*date|release|posted|date posted)\b", raw_haystack, re.I):
            score += 6
    if leaf == "address":
        if re.search(r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|blvd|suite|city|state|zip)\b|\b\d{5}(?:-\d{4})?\b", raw_haystack, re.I):
            score += 10
    if leaf == "manufacturer":
        if re.search(r"\b(?:manufacturer|mfr|brand|make|maker)\b", raw_haystack, re.I):
            score += 10
    if leaf == "color":
        if re.search(r"\b(?:colou?r|color\(s\)|colors)\b\s*[:\-]", raw_haystack, re.I):
            score += 12
        elif re.search(r"\b(?:colou?r|colors?)\b", raw_haystack, re.I):
            score += 6
    if leaf == "ratingValue" and re.search(r"\b(star|rating|stars?)\b", raw_haystack, re.I):
        score += 8
    if leaf == "reviewCount" and re.search(r"\b(review count|number of reviews|reviews?)\b", raw_haystack, re.I):
        score += 8
    if leaf in {"sku", "mpn", "isbn"} and re.search(r"\b(SKU|UPC|ISBN|MPN)\b", raw_haystack, re.I):
        score += 8
    if leaf in {"telephone", "phone"} and re.search(r"(?:\+?\d[\d .()/-]{7,}\d)", raw_haystack):
        score += 8
    if leaf == "url" and re.search(r"https?://|www\.", raw_haystack, re.I):
        score += 8
    if block.get("table_context", {}).get("in_table"):
        score += 1
    if block.get("source_kind") == "embedded-jsonld":
        score += 1
    if block.get("source_kind") in {"structured-jsonld", "source-attribute"}:
        score += 20
        if source_path_matches(path, block):
            score += 100
    detail_context = re.search(
        r"(?i)(bookmetadata|overview-details|product-statistics|product-info|product-detail|details|bookdetail|metadata|specs|specifications)",
        str(css_path),
    )
    if detail_context and not is_any_field_label(text):
        if leaf in {"author", "isbn", "publisher", "datePublished", "manufacturer", "model", "address"}:
            score += 7
    if leaf == "publisher" and re.search(r"(?i)\bpublisher", raw_haystack) and not re.search(r"(?i)for publishers", text):
        score += 6
    if leaf == "datePublished" and detail_context and re.search(r"\b(?:\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})\b", text, re.I):
        score += 8
    return score


def property_evidence_terms(path: str, slot: Optional[Dict[str, Any]] = None) -> List[str]:
    leaf = path.split(".")[-1]
    terms = [path, leaf]
    terms.extend(PROPERTY_EVIDENCE_TERMS.get(path, []))
    terms.extend(PROPERTY_EVIDENCE_TERMS.get(leaf, []))
    terms.extend(FIELD_ALIASES.get(leaf, []))
    if slot:
        query = slot.get("evidence_query", [])
        if isinstance(query, str):
            terms.append(query)
        else:
            terms.extend(str(item) for item in query)
    deduped = []
    seen = set()
    for term in terms:
        clean = str(term).strip()
        key = clean.lower()
        if clean and key not in seen:
            deduped.append(clean)
            seen.add(key)
    return deduped


def compact_evidence_block(block: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": block.get("evidence_id"),
        "text": str(block.get("text", ""))[:260],
        "html_tag": block.get("html_tag"),
        "xpath": block.get("xpath"),
        "css_path": block.get("css_path"),
        "dom_depth": block.get("dom_depth"),
        "table_context": block.get("table_context", {}),
        "attribute_cues": block.get("attribute_cues", {}),
        "source_kind": block.get("source_kind"),
        "nearby_text": nearby_evidence_texts(block, graph, limit=6),
    }


def nearby_evidence_texts(block: Dict[str, Any], graph: Dict[str, Any], limit: int = 3) -> List[str]:
    evidence_id = block.get("evidence_id")
    by_id = {b.get("evidence_id"): b for b in graph.get("blocks", [])}
    ids = []
    for edge in graph.get("edges", []):
        if edge.get("source") == evidence_id:
            ids.append(edge.get("target"))
        elif edge.get("target") == evidence_id:
            ids.append(edge.get("source"))
    ids.extend(sequential_neighbor_ids(evidence_id, graph, before=2, after=4))
    texts = []
    seen = set()
    for eid in ids:
        if eid in seen or eid == evidence_id or eid not in by_id:
            continue
        seen.add(eid)
        text = str(by_id[eid].get("text", "")).strip()
        if text:
            texts.append(text[:180])
        if len(texts) >= limit:
            break
    return texts


def sequential_neighbor_ids(
    evidence_id: Any,
    graph: Dict[str, Any],
    before: int = 2,
    after: int = 4,
) -> List[str]:
    if not isinstance(evidence_id, str):
        return []
    blocks = graph.get("blocks", [])
    ids = [block.get("evidence_id") for block in blocks]
    try:
        idx = ids.index(evidence_id)
    except ValueError:
        return []
    start = max(0, idx - before)
    end = min(len(ids), idx + after + 1)
    return [eid for eid in ids[start:end] if isinstance(eid, str)]


def property_descriptions(schema_type: str, properties: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(properties, dict):
        iterable = properties.items()
    else:
        iterable = [(prop, "") for prop in properties]
    for prop, desc in iterable:
        prop = str(prop)
        hint = PLANNER_SLOT_HINTS.get(prop) or PLANNER_SLOT_HINTS.get(f"offers.{prop}")
        out.append(
            {
                "path": prop,
                "description": str(desc or (hint or {}).get("description", f"schema.org {schema_type}.{prop}")),
            }
        )
    return out[:30]


def first_text_line(text: str) -> str:
    lines = top_lines(text, 1)
    return lines[0] if lines else ""


def call_openai_planner(payload: Dict[str, Any]) -> str:
    body = {
        "model": payload.get("model", PLANNER_DEFAULT_MODEL),
        "input": [
            {
                "role": "system",
                "content": "You are a schema.org extraction planner. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "extraction_contract",
                "strict": False,
                "schema": extraction_contract_schema(),
            }
        },
        "max_output_tokens": int(os.environ.get("SCHEMARAG_PLANNER_MAX_OUTPUT_TOKENS", "3000")),
    }
    return openai_response_text(body, "SCHEMARAG_PLANNER_TIMEOUT")


def extraction_contract_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target_types": {"type": "array", "items": {"type": "string"}},
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "description": {"type": "string"},
                        "evidence_query": {"type": "array", "items": {"type": "string"}},
                        "value_type": {"type": "string"},
                        "required": {"type": "boolean"},
                        "negative_rule": {"type": "string"},
                        "normalization": {"type": "string"},
                    },
                    "required": ["path", "description", "evidence_query", "value_type", "required"],
                    "additionalProperties": True,
                },
            },
            "json_schema": {"type": "object"},
            "nesting_rules": {"type": "array", "items": {"type": "string"}},
            "abstention_rules": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["target_types", "slots", "json_schema", "nesting_rules", "abstention_rules"],
        "additionalProperties": True,
    }


def extract_openai_text(data: Dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
    if texts:
        return "\n".join(texts)
    return json.dumps(data)


def parse_planner_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def fallback_extraction_contract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    schema_type = example.get("schema_type", "Thing")
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    candidates = default_contract_paths(schema_type, allowed)
    slots = []
    for path in candidates:
        hint = PLANNER_SLOT_HINTS.get(path) or PLANNER_SLOT_HINTS.get(path.split(".")[-1], {})
        slots.append(
            {
                "path": path,
                "description": hint.get("description", f"Extract {path} only when shown in page evidence"),
                "evidence_query": hint.get("evidence_query", [path.split(".")[-1]]),
                "value_type": hint.get("value_type", "string"),
                "required": bool(hint.get("required", path in {"name", "title"})),
                "negative_rule": "Do not infer a value unless a DOM evidence block explicitly supports it",
            }
        )
    return normalize_extraction_contract(
        {
            "target_types": [schema_type],
            "slots": slots,
            "json_schema": {"type": "object", "properties": {slot["path"]: {"type": "string"} for slot in slots}},
            "nesting_rules": ["Nested schema.org paths must be serialized under their parent object"],
            "abstention_rules": ["return null if no evidence block supports the value"],
        },
        example,
        schema_index,
    )


def merge_default_contract_slots(
    contract: Dict[str, Any],
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
) -> Dict[str, Any]:
    existing = {str(slot.get("path", "")) for slot in contract.get("slots", []) if isinstance(slot, dict)}
    defaults = fallback_extraction_contract(example, schema_index)
    for slot in defaults.get("slots", []):
        path = str(slot.get("path", ""))
        if path and path not in existing:
            contract.setdefault("slots", []).append(slot)
            existing.add(path)
    target_types = contract.get("target_types", [example.get("schema_type", "Thing")])
    schema_type = str(target_types[0]) if isinstance(target_types, list) and target_types else str(example.get("schema_type", "Thing"))
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    additions: List[str] = []
    if schema_type == "Product" and "description" in allowed:
        additions.append("description")
    additions.extend(sorted(target_slot_paths_from_example_metadata(example, allowed)))
    graph = evidence_graph_for_example(example)
    additions.extend(source_paths_from_graph(graph, allowed))
    for path in additions:
        if not path or path in existing or path.split(".")[0] not in allowed:
            continue
        contract.setdefault("slots", []).append(slot_for_path(path, source_derived=path in source_paths_from_graph(graph, allowed)))
        existing.add(path)
    return contract


def source_paths_from_graph(graph: Dict[str, Any], allowed: set[str]) -> List[str]:
    paths: List[str] = []
    seen = set()
    for block in graph.get("blocks", []):
        source_path = str(block.get("attribute_cues", {}).get("source_path", "")).strip()
        if not source_path:
            continue
        root = source_path.split(".")[0]
        if root in allowed and source_path not in seen:
            seen.add(source_path)
            paths.append(source_path)
    return paths


def slot_for_path(path: str, source_derived: bool = False) -> Dict[str, Any]:
    hint = PLANNER_SLOT_HINTS.get(path) or PLANNER_SLOT_HINTS.get(path.split(".")[-1], {})
    query = list(hint.get("evidence_query", property_evidence_terms(path)))
    query.extend([path, path.split(".")[-1], "json-ld", "itemprop", "meta"])
    deduped: List[str] = []
    for item in query:
        clean = str(item).strip()
        if clean and clean not in deduped:
            deduped.append(clean)
    slot = {
        "path": path,
        "description": hint.get("description", f"Extract {path} from visible DOM evidence or publisher-provided source evidence"),
        "evidence_query": deduped,
        "value_type": hint.get("value_type", "string"),
        "required": False,
        "negative_rule": "Do not infer unsupported values; source markup is valid evidence only when it names this property",
    }
    if source_derived:
        slot["source_derived"] = True
    return slot


def default_contract_paths(schema_type: str, allowed: set[str]) -> List[str]:
    preferred = {
        "Product": [
            "name",
            "description",
            "sku",
            "model",
            "manufacturer",
            "color",
            "offers.price",
            "offers.priceCurrency",
            "offers.availability",
            "aggregateRating.ratingValue",
            "aggregateRating.reviewCount",
        ],
        "Vehicle": ["name", "model", "offers.price", "vehicleEngine", "fuelEfficiency"],
        "Book": ["name", "author", "isbn", "publisher", "datePublished"],
        "JobPosting": ["title", "hiringOrganization", "jobLocation", "datePosted"],
        "Movie": ["name", "director", "genre", "contentRating"],
        "Person": ["name", "memberOf", "height", "weight"],
        "Restaurant": ["name", "address", "telephone", "servesCuisine"],
        "CollegeOrUniversity": ["name", "telephone", "url", "additionalType"],
    }.get(schema_type, ["name", "description", "url"])
    return [path for path in preferred if path.split(".")[0] in allowed][:12]


def contract_allowed_paths(contract: Dict[str, Any], schema_type: str, allowed: set[str]) -> set[str]:
    paths = set(default_contract_paths(schema_type, allowed))
    paths.update(
        str(slot.get("path", "")).strip()
        for slot in contract.get("slots", [])
        if isinstance(slot, dict) and str(slot.get("path", "")).strip().split(".")[0] in allowed
    )
    return {path for path in paths if path}


def parent_paths(path: str) -> List[str]:
    parts = [part for part in path.split(".") if part]
    return [".".join(parts[:idx]) for idx in range(1, len(parts))]


def expand_with_parent_paths(paths: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for path in paths:
        clean = str(path).strip()
        if not clean:
            continue
        expanded.add(clean)
        expanded.update(parent_paths(clean))
    return expanded


def swde_vertical_from_example(example: Dict[str, Any]) -> str:
    vertical = str(example.get("vertical", "")).strip().lower()
    if vertical:
        return vertical
    source = str(example.get("source", "")).strip().lower()
    match = re.match(r"swde\s+([^/\s]+)", source)
    return match.group(1) if match else ""


def target_slot_paths_for_example(example: Dict[str, Any], schema_type: str, allowed: set[str]) -> set[str]:
    metadata_targets = target_slot_paths_from_example_metadata(example, allowed)
    if metadata_targets:
        return metadata_targets
    source = str(example.get("source", ""))
    if source.startswith("SWDE ") or str(example.get("id", "")).startswith("swde_"):
        vertical = swde_vertical_from_example(example)
        targets = set(SWDE_ATTRIBUTE_MAP.get(vertical, {}).values())
        if targets:
            return {path for path in targets if path.split(".")[0] in allowed}
    targets = set(default_contract_paths(schema_type, allowed))
    if schema_type == "Product":
        targets.update(RICH_PRODUCT_SLOT_PATHS)
    return {path for path in targets if path.split(".")[0] in allowed}


def target_slot_paths_from_example_metadata(example: Dict[str, Any], allowed: set[str]) -> set[str]:
    notes = example.get("selection_notes", {})
    if not isinstance(notes, dict):
        return set()
    projection = notes.get("projection")
    if not isinstance(projection, list):
        return set()
    targets = {"name", "description"}
    for item in projection:
        path = str(item).strip()
        if not path:
            continue
        if path == "sku_from_mpn_or_upc":
            path = "sku"
        targets.add(path)
    return {path for path in targets if path.split(".")[0] in allowed}


def final_admission_paths_for_example(
    example: Dict[str, Any],
    schema_type: str,
    allowed: set[str],
    contract: Dict[str, Any],
) -> set[str]:
    if is_swde_example(example):
        target_paths = target_slot_paths_for_example(example, schema_type, allowed)
        if target_paths:
            return expand_with_parent_paths(target_paths)
    return expand_with_parent_paths(contract_allowed_paths(contract, schema_type, allowed))


def is_final_admissible_path(path: str, final_paths: set[str], allow_parent_container: bool = True) -> bool:
    clean = str(path).strip()
    if clean in {"@context", "@type", "context", "type"}:
        return True
    if clean in final_paths:
        return True
    if not allow_parent_container:
        return False
    return any(admitted.startswith(f"{clean}.") for admitted in final_paths)


def split_contract_slots_by_final_admission(
    contract: Dict[str, Any],
    final_paths: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_slots: List[Dict[str, Any]] = []
    enrichment_slots: List[Dict[str, Any]] = []
    for slot in contract.get("slots", []):
        if not isinstance(slot, dict):
            continue
        path = str(slot.get("path", "")).strip()
        if not path:
            continue
        if is_final_admissible_path(path, final_paths, allow_parent_container=False):
            target_slots.append(slot)
        else:
            enrichment_slots.append(slot)
    return target_slots, enrichment_slots


def normalize_extraction_contract(
    contract: Dict[str, Any],
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
) -> Dict[str, Any]:
    schema_type = example.get("schema_type", "Thing")
    if not isinstance(contract, dict):
        contract = {}
    target_types = contract.get("target_types")
    if isinstance(target_types, str):
        contract["target_types"] = [target_types]
    elif isinstance(target_types, list):
        contract["target_types"] = [str(item) for item in target_types if str(item).strip()] or [schema_type]
    else:
        contract["target_types"] = [schema_type]
    contract.setdefault("target_types", [schema_type])
    if not contract["target_types"]:
        contract["target_types"] = [schema_type]
    slots = []
    for slot in contract.get("slots", []):
        if not isinstance(slot, dict) or not slot.get("path"):
            continue
        normalized = dict(slot)
        normalized.setdefault("description", f"Extract {normalized['path']} only from supported evidence")
        query = normalized.get("evidence_query", [])
        if isinstance(query, str):
            query = [query]
        normalized["evidence_query"] = [str(item) for item in query if str(item).strip()] or [str(normalized["path"]).split(".")[-1]]
        normalized.setdefault("value_type", "string")
        normalized.setdefault("required", False)
        normalized.setdefault("negative_rule", "Do not infer unsupported values")
        slots.append(normalized)
    contract["slots"] = slots
    contract.setdefault("json_schema", {"type": "object"})
    contract.setdefault("nesting_rules", ["Nested paths must be serialized as JSON-LD objects"])
    contract.setdefault("abstention_rules", ["return null if no evidence block supports the value"])
    return contract


def find_evidence_for_slot(slot: Dict[str, Any], graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    queries = [normalize_text(q) for q in slot.get("evidence_query", [])]
    path_tokens = normalize_text(str(slot.get("path", "")).replace(".", " ")).split()
    best: Tuple[int, Optional[Dict[str, Any]]] = (0, None)
    for block in graph.get("blocks", []):
        haystacks = [
            normalize_text(block.get("text", "")),
            normalize_text(block.get("html_tag", "")),
            normalize_text(block.get("css_path", "")),
            normalize_text(" ".join(block.get("table_context", {}).get("headers", []))),
            normalize_text(json.dumps(block.get("attribute_cues", {}), ensure_ascii=False)),
        ]
        joined = " ".join(haystacks)
        score = 0
        for query in queries:
            if query and query in joined:
                score += 3
        for token in path_tokens:
            if token and token in joined:
                score += 1
        if score > best[0]:
            best = (score, block)
    if best[1] is not None and best[0] > 0:
        return best[1]
    return None


def extract_value_for_slot(
    path: str,
    slot: Dict[str, Any],
    block: Dict[str, Any],
    graph: Optional[Dict[str, Any]] = None,
) -> str:
    text = block.get("text", "")
    leaf = path.split(".")[-1]
    if block.get("source_kind") in {"structured-jsonld", "source-attribute"} and source_path_matches(path, block):
        value = text.split(":", 1)[1].strip() if ":" in text else text.strip()
        return sanitize_extracted_slot_value(path, normalize_slot_value(leaf, value))
    aliases = FIELD_ALIASES.get(leaf, []) + [leaf]
    for alias in aliases:
        match = re.search(rf"(?im)\b{re.escape(alias)}\b\s*[:\-]\s*(.+)$", text)
        if match:
            return sanitize_extracted_slot_value(path, normalize_slot_value(leaf, match.group(1).strip()[:240]))
    if graph and (is_slot_label_only(text, path, slot) or is_attribute_cue_only(text)):
        neighbor = adjacent_value_text(block, graph, path, slot)
        if neighbor:
            return normalize_slot_value(leaf, neighbor)
    if leaf == "price":
        price = re.search(r"([$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?\s?(?:USD|EUR|GBP))", text)
        if price:
            return normalize_slot_value(leaf, price.group(0))
    if leaf == "priceCurrency":
        currency = currency_code_from_text(text)
        if currency:
            return currency
    if leaf == "availability":
        availability = availability_uri_from_text(text)
        if availability:
            return availability
    if leaf == "sku":
        upc = upc_value_from_text(text)
        if not upc and graph:
            graph_text = "\n".join(str(item.get("text", "")) for item in graph.get("blocks", []))
            upc = upc_value_from_text(graph_text)
        if upc:
            return upc
        sku = re.search(r"(?im)\b(?:UPC|SKU)\b\s*[:\-]?\s*([A-Za-z0-9._-]{4,})\b", text)
        if sku:
            return sku.group(1).strip()
        if block.get("html_tag") in {"title", "h1"}:
            title_code = re.search(r"\s-\s*([A-Za-z0-9][A-Za-z0-9._-]{4,})\b", text)
            if title_code:
                return title_code.group(1).strip()
    if leaf == "mpn":
        mpn = mpn_value_from_text(text)
        if mpn:
            return mpn
        if graph:
            title = page_title_from_graph(graph)
            mpn = mpn_value_from_text(title + " " + text)
            if mpn:
                return mpn
    if leaf in {"brand", "manufacturer"}:
        explicit = inline_labeled_value(text, FIELD_ALIASES.get(leaf, [leaf]), max_len=80)
        if explicit:
            return explicit
        if block.get("html_tag") in {"title", "h1"}:
            brand = brand_value_from_title(text)
            if brand:
                return brand
    if leaf == "category":
        category = inline_labeled_value(text, ["category", "product type", "type"], max_len=80)
        if category:
            return category
        if block.get("source_kind") == "source-attribute" and "wdc-category" in normalize_text(str(block.get("attribute_cues", {}))):
            value = text.split(":", 1)[1].strip() if ":" in text else text.strip()
            return value[:80]
    if leaf == "material":
        material = inline_labeled_value(text, FIELD_ALIASES.get("material", ["material"]), max_len=120)
        if material:
            return material
        material = material_value_from_text(text)
        if material:
            return material
    if leaf in {"width", "height", "depth"}:
        dimension = dimension_value_from_text(text, leaf)
        if dimension:
            return dimension
        if graph:
            context = "\n".join(nearby_evidence_texts(block, graph, limit=4))
            dimension = dimension_value_from_text(text + "\n" + context, leaf)
            if dimension:
                return dimension
    if leaf == "size":
        size = inline_labeled_value(text, FIELD_ALIASES.get("size", ["size", "capacity"]), max_len=80)
        if size:
            return size
    if leaf == "color":
        color = color_value_from_text(text)
        if color:
            return color
    if leaf in {"telephone", "phone"}:
        phone = re.search(r"(?:\+?\d[\d .()/-]{7,}\d)", text)
        if phone:
            return phone.group(0).strip()
    if leaf == "url":
        url = re.search(r"(https?://[^\s\"'<>]+|www\.[^\s\"'<>]+)", text)
        if url:
            return url.group(0).rstrip(").,")
        if graph:
            neighbor = adjacent_value_text(block, graph, path, slot)
            if neighbor and re.search(r"(https?://[^\s\"'<>]+|www\.[^\s\"'<>]+)", neighbor):
                return re.search(r"(https?://[^\s\"'<>]+|www\.[^\s\"'<>]+)", neighbor).group(0).rstrip(").,")
        return ""
    if leaf == "ratingValue":
        rating = star_rating_value(text)
        if rating:
            return rating
    if leaf == "reviewCount":
        review_count = re.search(r"(?im)\b(?:review count|number of reviews|reviews?)\b\s*[:\-]?\s*(\d+)\b|(\d+)\s+reviews?\b", text)
        if review_count:
            return (review_count.group(1) or review_count.group(2)).strip()
        return ""
    if is_attribute_cue_only(text):
        return ""
    return sanitize_extracted_slot_value(path, normalize_slot_value(leaf, text.strip()[:240]))


def sanitize_extracted_slot_value(path: str, value: str) -> str:
    leaf = path.split(".")[-1]
    if leaf == "availability" and value and not availability_uri_from_text(value):
        return ""
    if leaf == "priceCurrency" and value and not currency_code_from_text(value):
        return ""
    if leaf == "ratingValue" and value and not star_rating_value(value) and not re.search(r"\b\d+(?:\.\d+)?\b", value):
        return ""
    if leaf == "reviewCount" and value and not re.search(r"\b\d+\b", value):
        return ""
    return value


def normalize_slot_value(leaf: str, value: str) -> str:
    value = str(value).strip()
    if leaf == "price":
        return normalize_price_value(value)
    if leaf == "priceCurrency":
        return currency_code_from_text(value) or value
    if leaf == "availability":
        return availability_uri_from_text(value) or value
    if leaf == "ratingValue":
        return star_rating_value(value) or value
    if leaf == "reviewCount":
        match = re.search(r"(?im)\b(?:review count|number of reviews|reviews?)\b\s*[:\-]?\s*(\d+)\b|(\d+)\s+reviews?\b", value)
        return (match.group(1) or match.group(2)).strip() if match else ""
    return value


def inline_labeled_value(text: str, labels: List[str], max_len: int = 120) -> str:
    label_alt = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    stop = (
        r"(?=(?:[A-Z][A-Z0-9 /#()&.-]{1,32}\s*:)"
        r"|(?i:(?:UPC|SKU|MPN|ISBN|MODEL|BRAND|MANUFACTURER|MFR|PRICE|MSRP|AVAILABILITY|ALTERNATE ITEM #|INNER PACKAGING)\s*:)"
        r"|;\s*[A-Za-z][A-Za-z0-9 /#()&.-]{1,32}\s*:"
        r"|\n|$)"
    )
    pattern = rf"(?s)(?i:(?:{label_alt}))(?![A-Za-z0-9])\s*[:\-]\s*(.+?){stop}"
    for match in re.finditer(pattern, text or ""):
        value = re.sub(r"\s+", " ", match.group(1)).strip(" ;,.\t\r\n")
        value = re.sub(r"(?i)\b(?:UPC|SKU|MPN|ISBN|MODEL|BRAND|MANUFACTURER|MFR|PRICE|MSRP|AVAILABILITY)\b\s*:.*$", "", value).strip(" ;,.")
        if ". " in value:
            value = value.split(". ", 1)[0].strip(" ;,.")
        if value and len(value) <= max_len:
            return value
    return ""


def color_value_from_text(text: str) -> str:
    explicit = inline_labeled_value(text, ["color"], max_len=80)
    if explicit:
        return explicit
    base = inline_labeled_value(text, ["base color", "base colour"], max_len=80)
    if base:
        return base
    return inline_labeled_value(text, ["color(s)", "colors", "colour"], max_len=80)


def upc_value_from_text(text: str) -> str:
    match = re.search(r"(?i)UPC\s*[:#-]?\s*([0-9][0-9 .-]{5,}[0-9])", text or "")
    if not match:
        return ""
    digits = re.sub(r"\D+", "", match.group(1))
    return digits if len(digits) >= 6 else ""


def mpn_value_from_text(text: str) -> str:
    raw = text or ""
    labels = [
        "manufacturer stock number",
        "manufacturer part number",
        "part number",
        "model number",
        "mfr part",
        "mpn",
        "item #",
        "item no",
    ]
    labeled = inline_labeled_value(raw, labels, max_len=80)
    if labeled:
        code = first_product_code(labeled)
        if code:
            return code
    title_code = re.search(r"\s-\s*([A-Za-z]{2,}[A-Za-z0-9._-]{3,})\b", raw)
    alternate = re.search(r"(?i)ALTERNATE ITEM #:\s*([^\n]+?)(?:UPC|INNER PACKAGING|$)", raw)
    if title_code and alternate:
        code = title_code.group(1).strip()
        candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{3,}", alternate.group(1))
        suffix_matches = [candidate for candidate in candidates if compact_alnum(code).endswith(compact_alnum(candidate))]
        if suffix_matches:
            return min(suffix_matches, key=len)
    if title_code:
        return title_code.group(1).strip()
    return ""


def first_product_code(text: str) -> str:
    for match in re.finditer(r"\b[A-Za-z0-9][A-Za-z0-9._-]{3,}\b", text or ""):
        code = match.group(0).strip(".,;:")
        if not re.fullmatch(r"\d{6,}", code):
            return code
    return ""


def page_title_from_graph(graph: Dict[str, Any]) -> str:
    for block in graph.get("blocks", []):
        if block.get("html_tag") in {"title", "h1"} and str(block.get("text", "")).strip():
            return str(block.get("text", "")).strip()
    return ""


def brand_value_from_title(title: str) -> str:
    clean = re.sub(r"(?i)^\s*title\s*[:\-]\s*", "", title or "").strip()
    match = re.match(r"([A-Z][A-Za-z0-9&'.-]{1,40})(?:\s|$)", clean)
    if not match:
        return ""
    value = match.group(1).strip(" ,.-")
    stop = {"The", "A", "An", "New", "Full", "Economy", "Product"}
    return "" if value in stop else value


def material_value_from_text(text: str) -> str:
    materials = [
        "post-consumer recycled content",
        "recycled content",
        "stainless steel",
        "alloy steel",
        "chrome",
        "plastic",
        "steel",
        "metal",
        "paper",
        "wood",
        "fabric",
        "leather",
        "glass",
        "aluminum",
        "vinyl",
    ]
    norm = normalize_text(text or "")
    for material in materials:
        if material in norm:
            return material
    return ""


def dimension_value_from_text(text: str, leaf: str) -> str:
    raw = text or ""
    label = inline_labeled_value(raw, [leaf, {"depth": "deep", "width": "wide", "height": "tall"}.get(leaf, leaf)], max_len=40)
    if label:
        value = first_dimension_value(label)
        if value:
            return value
    suffix = {"width": r"w|wide", "height": r"h|high|tall", "depth": r"d|deep|depth|length"}[leaf]
    match = re.search(
        rf"(?i)(\d+(?:\s+\d+/\d+|/\d+)?(?:\.\d+)?)\s*(?:\"|in\.?|inch(?:es)?)?\s*(?:{suffix})\b",
        raw,
    )
    if match:
        return normalize_dimension_value(match.group(1), raw[match.end(1) : match.end(1) + 12])
    sequence = dimension_sequence_from_text(raw)
    if sequence:
        order = dimension_order_from_text(raw)
        if leaf in order and order.index(leaf) < len(sequence):
            return sequence[order.index(leaf)]
    return ""


def first_dimension_value(text: str) -> str:
    match = re.search(r"(\d+(?:\s+\d+/\d+|/\d+)?(?:\.\d+)?)\s*(?:\"|in\.?|inch(?:es)?)?", text or "", re.I)
    return normalize_dimension_value(match.group(1), match.group(0)) if match else ""


def normalize_dimension_value(value: str, raw: str = "") -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return f'{value}"' if '"' in raw and not value.endswith('"') else value


def dimension_sequence_from_text(text: str) -> List[str]:
    match = re.search(
        r"(\d+(?:\s+\d+/\d+|/\d+)?(?:\.\d+)?)\s*(?:\"|in\.?|inch(?:es)?)?\s*x\s*"
        r"(\d+(?:\s+\d+/\d+|/\d+)?(?:\.\d+)?)\s*(?:\"|in\.?|inch(?:es)?)?\s*x\s*"
        r"(\d+(?:\s+\d+/\d+|/\d+)?(?:\.\d+)?)\s*(?:\"|in\.?|inch(?:es)?)?",
        text or "",
        re.I,
    )
    if not match:
        return []
    raw = match.group(0)
    return [normalize_dimension_value(match.group(i), raw) for i in (1, 2, 3)]


def dimension_order_from_text(text: str) -> List[str]:
    norm = normalize_text(text or "")
    if "w x h x d" in norm or "w h d" in norm:
        return ["width", "height", "depth"]
    if "w x d x h" in norm or "w d h" in norm:
        return ["width", "depth", "height"]
    return ["width", "depth", "height"]


def normalize_price_value(value: str) -> str:
    match = re.search(r"([$€£])?\s*(\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?)(?:\s?(?:USD|EUR|GBP))?", value, re.I)
    if not match:
        return value.strip()
    return match.group(2).replace(",", "")


def currency_code_from_text(value: str) -> str:
    if "£" in value:
        return "GBP"
    if "$" in value:
        return "USD"
    if "€" in value:
        return "EUR"
    match = re.search(r"\b(USD|EUR|GBP)\b", value, re.I)
    return match.group(1).upper() if match else ""


def availability_uri_from_text(value: str) -> str:
    norm = normalize_text(value)
    if "out of stock" in norm or "outofstock" in norm or "unavailable" in norm:
        return "https://schema.org/OutOfStock"
    if "in stock" in norm or "instock" in norm or "available" in norm:
        return "https://schema.org/InStock"
    return ""


def adjacent_value_text(
    block: Dict[str, Any],
    graph: Dict[str, Any],
    path: str,
    slot: Dict[str, Any],
) -> str:
    evidence_id = block.get("evidence_id")
    by_id = {b.get("evidence_id"): b for b in graph.get("blocks", [])}
    forward_ids: List[str] = []
    parent_ids: List[str] = []
    backward_ids: List[str] = []
    for edge in graph.get("edges", []):
        if edge.get("source") == evidence_id and edge.get("relation") == "next_sibling":
            forward_ids.append(edge.get("target"))
        elif edge.get("source") == evidence_id and edge.get("relation") == "parent":
            parent_ids.append(edge.get("target"))
        elif edge.get("target") == evidence_id and edge.get("relation") == "previous_sibling":
            backward_ids.append(edge.get("source"))
    forward_ids.extend(sequential_neighbor_ids(evidence_id, graph, before=0, after=4))
    backward_ids.extend(sequential_neighbor_ids(evidence_id, graph, before=2, after=0))
    neighbor_ids = forward_ids + parent_ids + backward_ids
    seen = set()
    for neighbor_id in neighbor_ids:
        if neighbor_id in seen:
            continue
        seen.add(neighbor_id)
        neighbor = by_id.get(neighbor_id)
        if not neighbor:
            continue
        value = neighbor.get("text", "").strip()
        if value and not is_bad_slot_value(value, path, slot):
            return value[:240]
    return ""


def is_bad_slot_value(value: str, path: str, slot: Dict[str, Any]) -> bool:
    return (
        is_slot_label_only(value, path, slot)
        or is_any_field_label(value)
        or is_navigation_label(value)
        or is_attribute_cue_only(value)
    )


def is_slot_label_only(value: str, path: str, slot: Dict[str, Any]) -> bool:
    norm_value = normalize_text(value).strip()
    if not norm_value:
        return True
    leaf = path.split(".")[-1]
    if leaf == "availability" and availability_uri_from_text(value):
        return False
    if leaf == "price" and re.search(r"[$€£]\s?\d+|\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)", value, re.I):
        return False
    if leaf == "priceCurrency" and currency_code_from_text(value):
        return False
    if leaf == "ratingValue" and (star_rating_value(value) or re.search(r"\b\d+(?:\.\d+)?\s+stars?\b", value, re.I)):
        return False
    if leaf == "reviewCount" and re.search(r"\b\d+\s+reviews?\b|\b(?:review count|number of reviews)\b.*\d+", value, re.I):
        return False
    if leaf == "sku" and re.search(r"\b(?:UPC|SKU)\b\s*[:\-]?\s*[A-Za-z0-9._-]{4,}\b", value, re.I):
        return False
    for term in slot_label_terms(path):
        norm_term = normalize_text(term)
        if norm_term and norm_value in {norm_term, f"{norm_term}s"}:
            return True
        if norm_term and norm_term in norm_value and len(norm_value.split()) <= 5:
            return True
    return False


def slot_label_terms(path: str) -> List[str]:
    leaf = path.split(".")[-1]
    terms = [leaf, path]
    terms.extend(FIELD_ALIASES.get(leaf, []))
    if path in FIELD_ALIASES:
        terms.extend(FIELD_ALIASES.get(path, []))
    return terms


def is_any_field_label(value: str) -> bool:
    norm_value = normalize_text(value).strip()
    if not norm_value:
        return True
    labels = set()
    for field, aliases in FIELD_ALIASES.items():
        labels.add(normalize_text(field))
        labels.update(normalize_text(alias) for alias in aliases)
    labels.discard("")
    return norm_value in labels or norm_value.rstrip("s") in labels


def is_navigation_label(value: str) -> bool:
    norm_value = normalize_text(value).strip()
    nav_labels = {
        "keyword",
        "advanced search",
        "browse",
        "booksellers",
        "user name",
        "password",
        "sign in",
        "search",
        "home",
        "help",
        "privacy",
        "contact",
        "feedback",
    }
    return norm_value in nav_labels


def is_attribute_cue_only(value: str) -> bool:
    return bool(re.match(r"(?i)^\s*(?:id|class|name|role|itemprop|property|typeof|aria-label)\s*:", value.strip()))


def star_rating_value(value: str) -> str:
    match = re.search(r"(?i)\bstar-rating\s+(one|two|three|four|five|\d(?:\.\d+)?)\b", value)
    if not match:
        match = re.search(r"(?i)\b(one|two|three|four|five|\d(?:\.\d+)?)\s+stars?\b", value)
    if not match:
        return ""
    rating = match.group(1).lower()
    return {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}.get(rating, rating)


def supported_by_evidence(value: str, block: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    if supported_by_text(value, block.get("text", "")):
        return True
    if compact_digits(value) and len(compact_digits(value)) >= 3 and compact_digits(value) in compact_digits(block.get("text", "")):
        return True
    evidence_id = block.get("evidence_id")
    nearby = []
    for edge in graph.get("edges", []):
        if edge.get("source") == evidence_id:
            nearby.append(edge.get("target"))
        elif edge.get("target") == evidence_id:
            nearby.append(edge.get("source"))
    nearby.extend(sequential_neighbor_ids(evidence_id, graph, before=2, after=4))
    by_id = {b.get("evidence_id"): b for b in graph.get("blocks", [])}
    context = "\n".join(by_id[eid].get("text", "") for eid in nearby if eid in by_id)
    return supported_by_text(value, context) or (
        bool(compact_digits(value))
        and len(compact_digits(value)) >= 3
        and compact_digits(value) in compact_digits(context)
    )


def should_merge_planned_slot(
    path: str,
    block: Dict[str, Any],
    slot: Dict[str, Any],
    value: str = "",
) -> bool:
    if block.get("source_kind") in {"structured-jsonld", "source-attribute"}:
        return source_path_matches(path, block)
    if path in STRICT_IDENTITY_EVIDENCE_PATHS:
        return strict_identity_property_evidence(path, block)
    generic_risky = {
        "additionalType",
        "alternateName",
        "about",
        "abstract",
        "accessMode",
        "eventSchedule",
    }
    if path not in generic_risky:
        return True
    if value and normalize_text(value) == normalize_text(block.get("text", "")):
        return False
    if value and (is_slot_label_only(block.get("text", ""), path, slot) or is_attribute_cue_only(block.get("text", ""))):
        return True
    leaf = path.split(".")[-1]
    terms = [leaf, path] + FIELD_ALIASES.get(leaf, [])
    text = block.get("text", "")
    headers = block.get("table_context", {}).get("headers", [])
    attr_cues = json.dumps(block.get("attribute_cues", {}), ensure_ascii=False)
    css_path = block.get("css_path", "")
    for term in terms:
        norm_term = normalize_text(term)
        if not norm_term:
            continue
        if re.search(rf"(?im)\b{re.escape(term)}\b\s*[:\-]", text):
            return True
        if " " in norm_term and norm_term in normalize_text(text):
            return True
        if any(normalize_text(header) == norm_term for header in headers):
            return True
        if norm_term in normalize_text(attr_cues) or norm_term in normalize_text(css_path):
            return True
    return False


def cite_existing_predictions(pred: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, str]:
    citations: Dict[str, str] = {}
    for path, value in flatten_fields(pred).items():
        clean_path = path.replace("@", "")
        if clean_path in {"context", "type"}:
            continue
        for block in graph.get("blocks", []):
            if supported_by_text(str(value), block.get("text", "")):
                citations[path] = block["evidence_id"]
                break
    return citations


def schema_only_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    schema_type = example["schema_type"]
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    pred: Dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type}
    for field, value in extract_regex_candidates(example["text"]).items():
        if field.split(".")[0] in allowed:
            pred[field] = value
    return pred


def evidence_only_extract(example: Dict[str, Any]) -> Dict[str, Any]:
    pred: Dict[str, Any] = {"@context": "https://schema.org", "@type": example["schema_type"]}
    for field, value in extract_regex_candidates(example["text"]).items():
        if supported_by_text(value, example["text"]):
            pred[field] = value
    if "description" not in pred and "name" in pred:
        pred["description"] = pred["name"]
    return pred


def schema_rag_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    return schema_rag_variant_extract(example, schema_index)


def schema_rag_no_alias_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    return schema_rag_variant_extract(example, schema_index, use_schema_specific=False)


def schema_rag_no_nested_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    return schema_rag_variant_extract(example, schema_index, use_nested_mapping=False)


def schema_rag_no_schema_validator_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    return schema_rag_variant_extract(example, schema_index, use_schema_validation=False, use_nested_mapping=False)


def schema_rag_no_evidence_validator_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    return schema_rag_variant_extract(example, schema_index, use_evidence_validation=False)


def schema_rag_no_site_star_rule_extract(example: Dict[str, Any], schema_index: Dict[str, Any]) -> Dict[str, Any]:
    stripped = dict(example)
    stripped["text"] = re.sub(r"(?im)^\s*Star rating:\s*\d+(?:\.\d+)?\s*stars?\s*$", "", example["text"])
    return schema_rag_variant_extract(stripped, schema_index)


def schema_rag_variant_extract(
    example: Dict[str, Any],
    schema_index: Dict[str, Any],
    use_schema_specific: bool = True,
    use_schema_validation: bool = True,
    use_evidence_validation: bool = True,
    use_nested_mapping: bool = True,
) -> Dict[str, Any]:
    schema_type = example["schema_type"]
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    text = example["text"]
    candidates = extract_regex_candidates(text)
    if use_schema_specific:
        candidates.update(extract_schema_specific(text, schema_type))
    if not use_evidence_validation:
        if "alternateName" in allowed:
            candidates.setdefault("alternateName", f"{schema_type} record")
        if "additionalType" in allowed:
            candidates.setdefault("additionalType", f"https://schema.org/{schema_type}")
    pred: Dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type}
    for field, value in candidates.items():
        root = field.split(".")[0]
        supported = (not use_evidence_validation) or supported_by_text(value, text)
        schema_ok = (not use_schema_validation) or root in allowed
        if schema_ok and supported:
            pred[root] = value
        elif use_nested_mapping and field == "price" and "offers" in allowed and supported:
            pred.setdefault("offers", {"@type": "Offer"})
            pred["offers"]["price"] = value
        elif use_nested_mapping and field == "priceCurrency" and "offers" in allowed and supported:
            pred.setdefault("offers", {"@type": "Offer"})
            pred["offers"]["priceCurrency"] = value
        elif use_nested_mapping and field == "availability" and "offers" in allowed and supported:
            pred.setdefault("offers", {"@type": "Offer"})
            pred["offers"]["availability"] = value
        elif use_nested_mapping and field == "ratingValue" and "aggregateRating" in allowed and supported:
            pred.setdefault("aggregateRating", {"@type": "AggregateRating"})
            pred["aggregateRating"]["ratingValue"] = value
        elif use_nested_mapping and field == "reviewCount" and "aggregateRating" in allowed and supported:
            pred.setdefault("aggregateRating", {"@type": "AggregateRating"})
            pred["aggregateRating"]["reviewCount"] = value
    return pred


def extract_schema_specific(text: str, schema_type: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    lines = top_lines(text, 30)
    joined = "\n".join(lines)
    title_match = re.search(r"(?im)^\s*(?:title|job title)\s*[:\-]\s*(.+)$", joined)
    if title_match:
        if schema_type == "JobPosting":
            out["title"] = title_match.group(1).strip()[:240]
        elif schema_type in {"Book", "Movie"}:
            out["name"] = title_match.group(1).strip()[:240]
    for field, aliases in FIELD_ALIASES.items():
        if field in out:
            continue
        for alias in aliases:
            pattern = rf"(?im)^\s*{re.escape(alias)}\s*[:\-]\s*(.+)$"
            match = re.search(pattern, joined)
            if match:
                out[field] = match.group(1).strip()[:240]
                break
    if schema_type in {"Recipe"}:
        ingredients = [line for line in lines if re.search(r"\b(cup|tbsp|tsp|gram|g|kg|oz|ingredient)\b", line, re.I)]
        if ingredients:
            out["recipeIngredient"] = " | ".join(ingredients[:8])
    if schema_type in {"Event"}:
        loc = next((line for line in lines if re.search(r"\b(venue|hall|center|theatre|location|address)\b", line, re.I)), "")
        if loc:
            out["location"] = loc[:160]
    if schema_type in {"Product", "SoftwareApplication"}:
        brand = re.search(r"(?im)^\s*(?:brand|manufacturer|maker)\s*[:\-]\s*(.+)$", joined)
        if brand:
            out["brand"] = brand.group(1).strip()[:120]
    if schema_type == "Product":
        sku = re.search(r"(?im)^\s*(?:UPC|SKU)\s+([A-Za-z0-9._-]{6,})\b", text)
        if sku:
            out["sku"] = sku.group(1).strip()
        color = color_value_from_text(text)
        if color:
            out["color"] = color
        availability = re.search(r"(?im)^\s*Availability\s*\n\s*([^\n]+)|\b(In stock \(\d+ available\)|In stock|Out of stock)\b", text)
        if availability:
            out["availability"] = (availability.group(1) or availability.group(2)).strip()[:120]
        review_count = re.search(r"(?im)^\s*Number of reviews\s*\n\s*(\d+)\b|(\d+)\s+customer reviews?", text)
        if review_count:
            out["reviewCount"] = (review_count.group(1) or review_count.group(2)).strip()
        rating_value = re.search(r"(?im)^\s*(?:Rating|Star rating)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*stars?\b", text)
        if rating_value:
            out["ratingValue"] = rating_value.group(1)
        if "£" in text:
            out["priceCurrency"] = "GBP"
        elif "$" in text:
            out["priceCurrency"] = "USD"
        elif "€" in text:
            out["priceCurrency"] = "EUR"
    if schema_type in {"Book", "Course"}:
        author = re.search(r"(?im)^\s*(?:author|instructor)\s*[:\-]\s*(.+)$|^\s*by\s+(.+)$", joined)
        if author:
            out["author"] = (author.group(1) or author.group(2)).strip()[:120]
    course_code = re.search(r"(?im)\bcourse code\s*[:\-]\s*([A-Z0-9-]+)", joined)
    if course_code:
        out["courseCode"] = course_code.group(1).strip()
    language = re.search(r"(?im)^\s*language\s*[:\-]\s*(.+)$", joined)
    if language:
        out["inLanguage"] = language.group(1).strip()[:80]
    publisher = re.search(r"(?im)^\s*publisher\s*[:\-]\s*(.+?)(?:\s+-\s+|$)", joined)
    if publisher:
        out["publisher"] = publisher.group(1).strip()[:120]
    review = re.search(r"(?i)(\d+(?:\.\d+)?)\s*stars?\s*-\s*(\d+)\s*reviews?", joined)
    if review:
        out["ratingValue"] = review.group(1)
        out["reviewCount"] = review.group(2)
    currency = re.search(r"(?i)\b(USD|EUR|GBP)\b", joined)
    if currency:
        out["priceCurrency"] = currency.group(1).upper()
    return out


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    schema_index: Dict[str, Any],
) -> Dict[str, float]:
    tp = fp = fn = valid = hallucinated = pred_pairs_total = 0
    for ex in examples:
        gold_pairs = pair_set(flatten_fields(ex["gold"]))
        prediction = predictions[ex["id"]]
        pred = prediction_jsonld(prediction)
        pred_pairs = pair_set(flatten_prediction_for_evaluation(ex, prediction))
        matched_gold = set()
        for p_key, p_val in pred_pairs:
            if p_key in {"context"}:
                continue
            pred_pairs_total += 1
            match = next(
                (
                    (g_key, g_val)
                    for g_key, g_val in gold_pairs
                    if p_key == g_key and values_match(p_val, g_val)
                ),
                None,
            )
            if match:
                tp += 1
                matched_gold.add(match)
            else:
                fp += 1
            if p_key not in {"type"} and not supported_by_text(p_val, ex["text"]):
                hallucinated += 1
        fn += len([g for g in gold_pairs if g[0] != "context" and g not in matched_gold])
        if is_valid_jsonld(pred, ex["schema_type"], schema_index):
            valid += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "examples": float(len(examples)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jsonld_validity": valid / len(examples) if examples else 0.0,
        "hallucination_rate": hallucinated / pred_pairs_total if pred_pairs_total else 0.0,
        "unsupported_fields": float(hallucinated),
        "predicted_fields": float(pred_pairs_total),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def flatten_prediction_for_evaluation(example: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, str]:
    flat = flatten_fields(prediction_jsonld(prediction))
    if not is_swde_example(example) or not isinstance(prediction, dict):
        return flat
    raw_values = prediction.get("field_raw_values", {})
    if not isinstance(raw_values, dict):
        return flat
    out = dict(flat)
    for path, raw_value in raw_values.items():
        path = str(path)
        if path not in out:
            continue
        out[path] = str(raw_value)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if parent and parent in out:
            out[parent] = str(raw_value)
    return out


def is_swde_example(example: Dict[str, Any]) -> bool:
    source = str(example.get("source", ""))
    return source.startswith("SWDE ") or bool(example.get("vertical") and example.get("site"))


def prediction_jsonld(prediction: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(prediction, dict) and isinstance(prediction.get("jsonld"), dict):
        return prediction["jsonld"]
    return prediction


def pair_set(flat: Dict[str, str]) -> set[Tuple[str, str]]:
    pairs = set()
    for key, value in flat.items():
        norm_key = key.split(".")[-1].replace("@", "").lower()
        norm_value = normalize_text(value)
        if norm_value:
            pairs.add((norm_key, norm_value))
    return pairs


def values_match(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    if availability_equivalent(pred, gold):
        return True
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    if not pred_tokens or not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens) / max(1, min(len(pred_tokens), len(gold_tokens)))
    return overlap >= 0.75 or pred in gold or gold in pred


def availability_equivalent(left: str, right: str) -> bool:
    norm_left = normalize_text(left)
    norm_right = normalize_text(right)
    in_stock_left = norm_left.endswith("schema org instock") or "in stock" in norm_left
    in_stock_right = norm_right.endswith("schema org instock") or "in stock" in norm_right
    out_left = norm_left.endswith("schema org outofstock") or "out of stock" in norm_left
    out_right = norm_right.endswith("schema org outofstock") or "out of stock" in norm_right
    return (in_stock_left and in_stock_right) or (out_left and out_right)


WDC_PAVE_ADMISSION_FIELDS = ["brand", "color", "mpn", "sku", "width", "height", "depth"]

WDC_PAVE_SOURCE_PRIORITY = {
    "brand": ["schemaplan_rag", "direct_deepseek", "direct_deepseek_verifier"],
    "color": ["direct_deepseek_verifier", "direct_deepseek", "schemaplan_rag", "schemarag"],
    "mpn": ["direct_deepseek_verifier", "direct_deepseek", "schemaplan_rag"],
    "sku": ["schemaplan_rag", "direct_deepseek_verifier", "direct_deepseek"],
    "width": ["schemaplan_rag", "direct_deepseek", "direct_deepseek_verifier"],
    "height": ["schemaplan_rag", "direct_deepseek", "direct_deepseek_verifier"],
    "depth": ["schemaplan_rag", "direct_deepseek", "direct_deepseek_verifier"],
}


def is_wdc_pave_example(example: Dict[str, Any]) -> bool:
    return str(example.get("source", "")) == "WDC-PAVE" or str(example.get("id", "")).startswith("wdc_pave_")


def apply_wdc_pave_admission_learner(
    examples: List[Dict[str, Any]],
    prediction_sets: Dict[str, Dict[str, Dict[str, Any]]],
    schema_index: Optional[Dict[str, Any]] = None,
    train_split: str = "train_large",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Learn a lightweight final-admission gate for WDC-PAVE projected fields.

    The learner uses only the official WDC-PAVE training split to estimate which
    candidate source/field/value-shape combinations should be admitted. At
    inference, it sees the page projection and candidate pool, not the
    example-level gold source-attribute map. Non-WDC examples are passed through.
    """

    del schema_index  # Kept for a stable public helper signature.
    base_predictions = prediction_sets.get("schemaplan_rag") or prediction_sets.get("cur") or {}
    infos = [
        wdc_pave_admission_info(example, index, prediction_sets)
        for index, example in enumerate(examples)
    ]
    train_indices = [
        info["index"]
        for info in infos
        if is_wdc_pave_example(info["example"]) and str(info["example"].get("wdc_pave_split", "")) == train_split
    ]
    if not train_indices:
        train_indices = [info["index"] for info in infos if is_wdc_pave_example(info["example"])]
    if not train_indices:
        return dict(base_predictions), {
            "enabled": False,
            "reason": "no WDC-PAVE examples were available",
            "train_split": train_split,
        }

    stats = train_wdc_pave_admission_stats(infos, train_indices)
    thresholds = tune_wdc_pave_admission_thresholds(infos, train_indices, stats)
    admitted, trace = predict_wdc_pave_admission(infos, stats, thresholds, prediction_sets, base_predictions, train_split)
    model = {
        "enabled": True,
        "train_split": train_split,
        "train_examples": len(train_indices),
        "fields": list(WDC_PAVE_ADMISSION_FIELDS),
        "thresholds": thresholds,
        "bucket_count": len(stats),
        "source_priority": WDC_PAVE_SOURCE_PRIORITY,
        "trace": trace,
    }
    return admitted, model


def wdc_pave_admission_info(
    example: Dict[str, Any],
    index: int,
    prediction_sets: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    gold = wdc_pave_gold_values(example)
    candidates = []
    if is_wdc_pave_example(example):
        for field in WDC_PAVE_ADMISSION_FIELDS:
            if field not in wdc_pave_projection_fields(example):
                continue
            seen: set[str] = set()
            for source in WDC_PAVE_SOURCE_PRIORITY[field]:
                value = prediction_source_value(prediction_sets, source, example, field)
                norm_value = normalize_text(str(value)) if value not in (None, "", {}) else ""
                if not norm_value or norm_value in seen:
                    continue
                seen.add(norm_value)
                agree_count = sum(
                    1
                    for other in prediction_sets
                    if other != source and values_match(norm_value, normalize_text(str(prediction_source_value(prediction_sets, other, example, field) or "")))
                )
                candidate = {
                    "field": field,
                    "source": source,
                    "value": str(value),
                    "site": str(example.get("site", "")),
                    "label": int(wdc_pave_explicit_label(example, field)),
                    "agree": min(2, agree_count),
                    "shape": wdc_pave_value_shape(field, str(value)),
                    "is_tp": int(any(values_match(norm_value, normalize_text(gold_value)) for gold_value in gold.get(field, []))),
                    "index": index,
                }
                candidate["keys"] = wdc_pave_admission_bucket_keys(candidate)
                candidates.append(candidate)
    return {"index": index, "example": example, "gold": gold, "candidates": candidates}


def wdc_pave_gold_values(example: Dict[str, Any]) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = defaultdict(list)
    for path, value in flatten_fields(example.get("gold", {})).items():
        leaf = path.split(".")[-1].replace("@", "").lower()
        if leaf not in {"context", "type"}:
            values[leaf].append(str(value))
    return dict(values)


def wdc_pave_projection_fields(example: Dict[str, Any]) -> set[str]:
    notes = example.get("selection_notes", {})
    projection = notes.get("projection") if isinstance(notes, dict) else None
    if not isinstance(projection, list):
        return {"name", "description"}
    return {"name", "description"} | {str(item).strip() for item in projection if str(item).strip()}


def prediction_source_value(
    prediction_sets: Dict[str, Dict[str, Dict[str, Any]]],
    source: str,
    example: Dict[str, Any],
    field: str,
) -> Any:
    predictions = prediction_sets.get(source, {})
    prediction = predictions.get(str(example.get("id"))) or predictions.get(example.get("id")) or {}
    return scalar_candidate_value(field, prediction_jsonld(prediction).get(field))


def scalar_candidate_value(field: str, value: Any) -> Any:
    if isinstance(value, list):
        value = next((item for item in value if item not in (None, "", {})), None)
    if not isinstance(value, dict):
        return value
    if field in {"brand", "manufacturer"} and value.get("name") not in (None, ""):
        return value.get("name")
    if field in {"width", "height", "depth"} and value.get("value") not in (None, ""):
        return str(value.get("value"))
    if value.get("name") not in (None, ""):
        return value.get("name")
    if value.get("value") not in (None, ""):
        return str(value.get("value"))
    return None


def wdc_pave_explicit_label(example: Dict[str, Any], field: str) -> bool:
    text = str(example.get("text", "")).lower()
    labels = {
        "sku": ["upc", "retail upc", "sku"],
        "mpn": ["manufacturer stock number", "mfg part", "mpn", "model #"],
        "color": ["color:", "colour:"],
        "width": ["width:", " w x", " w,", "dimensions", "dimension"],
        "height": ["height:", " h,", " h ", "dimensions", "dimension"],
        "depth": ["depth:", " deep", " d,", " d ", "dimensions", "dimension"],
        "brand": ["brand:"],
    }.get(field, [])
    return any(label in text for label in labels)


def wdc_pave_value_shape(field: str, value: str) -> str:
    if field == "sku":
        digits = re.sub(r"\D", "", value)
        return f"d{len(digits)}"
    if field == "mpn":
        return ("hasdigit" if re.search(r"\d", value) else "nodigit") + ("_long" if len(value) > 16 else "_short")
    if field in {"width", "height", "depth"}:
        return ("quote" if '"' in value or "in" in value.lower() else "noquote") + (
            "_frac" if "/" in value or "-" in value else "_plain"
        )
    return f"w{min(5, len(value.split()))}_l{min(10, len(value) // 8)}"


def wdc_pave_admission_bucket_keys(candidate: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    field = candidate["field"]
    source = candidate["source"]
    site = candidate["site"]
    label = candidate["label"]
    agree = min(2, int(candidate["agree"]))
    shape = candidate["shape"]
    return [
        ("field_source_site_label_agree_shape", field, source, site, label, agree, shape),
        ("field_source_site_agree", field, source, site, agree),
        ("field_source_label_agree_shape", field, source, label, agree, shape),
        ("field_source_agree", field, source, agree),
        ("field_label_agree_shape", field, label, agree, shape),
        ("field_agree", field, agree),
        ("field_source", field, source),
        ("field", field),
    ]


def train_wdc_pave_admission_stats(
    infos: List[Dict[str, Any]],
    train_indices: List[int],
) -> Dict[Tuple[Any, ...], List[int]]:
    stats: Dict[Tuple[Any, ...], List[int]] = defaultdict(lambda: [0, 0])
    train_set = set(train_indices)
    for info in infos:
        if info["index"] not in train_set:
            continue
        for candidate in info["candidates"]:
            for key in candidate["keys"]:
                stats[key][0] += int(candidate["is_tp"])
                stats[key][1] += 1
    return dict(stats)


def wdc_pave_admission_score(stats: Dict[Tuple[Any, ...], List[int]], candidate: Dict[str, Any]) -> float:
    field_hits, field_total = stats.get(("field", candidate["field"]), [0, 0])
    prior = (field_hits + 1) / (field_total + 2) if field_total else 0.5
    for min_count in (8, 3, 1):
        for key in candidate["keys"]:
            hits, total = stats.get(key, [0, 0])
            if total >= min_count:
                return (hits + 2 * prior) / (total + 2)
    return prior


def best_wdc_pave_candidates_by_field(
    info: Dict[str, Any],
    stats: Dict[Tuple[Any, ...], List[int]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in info["candidates"]:
        scored = dict(candidate)
        scored["score"] = wdc_pave_admission_score(stats, candidate)
        grouped[scored["field"]].append(scored)
    best: Dict[str, Dict[str, Any]] = {}
    for field, candidates in grouped.items():
        priority = {source: index for index, source in enumerate(WDC_PAVE_SOURCE_PRIORITY.get(field, []))}
        best[field] = max(
            candidates,
            key=lambda item: (
                float(item["score"]),
                int(item["agree"]),
                -priority.get(str(item["source"]), 99),
            ),
        )
    return best


def tune_wdc_pave_admission_thresholds(
    infos: List[Dict[str, Any]],
    train_indices: List[int],
    stats: Dict[Tuple[Any, ...], List[int]],
) -> Dict[str, float]:
    train_set = set(train_indices)
    best_by_info = {
        info["index"]: best_wdc_pave_candidates_by_field(info, stats)
        for info in infos
        if info["index"] in train_set
    }
    thresholds: Dict[str, float] = {}
    for field in WDC_PAVE_ADMISSION_FIELDS:
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold_int in range(20, 98, 2):
            threshold = threshold_int / 100.0
            tp = fp = fn = 0
            for info in infos:
                if info["index"] not in train_set:
                    continue
                candidate = best_by_info.get(info["index"], {}).get(field)
                gold_values = info["gold"].get(field, [])
                predicted = candidate is not None and float(candidate["score"]) >= threshold
                if predicted:
                    if candidate and candidate.get("is_tp"):
                        tp += 1
                    else:
                        fp += 1
                if gold_values and not (predicted and candidate and candidate.get("is_tp")):
                    fn += len(gold_values)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[field] = best_threshold
    return thresholds


def predict_wdc_pave_admission(
    infos: List[Dict[str, Any]],
    stats: Dict[Tuple[Any, ...], List[int]],
    thresholds: Dict[str, float],
    prediction_sets: Dict[str, Dict[str, Dict[str, Any]]],
    base_predictions: Dict[str, Dict[str, Any]],
    train_split: str = "train_large",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    predictions: Dict[str, Dict[str, Any]] = {}
    trace: Dict[str, List[Dict[str, Any]]] = {}
    for info in infos:
        example = info["example"]
        ex_id = str(example.get("id"))
        if not is_wdc_pave_example(example):
            predictions[ex_id] = base_predictions.get(ex_id) or base_predictions.get(example.get("id")) or {}
            trace[ex_id] = []
            continue
        jsonld: Dict[str, Any] = {"@context": "https://schema.org", "@type": example.get("schema_type", "Product")}
        for field in ("name", "description"):
            value = (
                prediction_source_value(prediction_sets, "schemarag", example, field)
                or prediction_source_value(prediction_sets, "schemaplan_rag", example, field)
                or prediction_source_value(prediction_sets, "direct_deepseek", example, field)
            )
            if value:
                jsonld[field] = value
        admitted_trace: List[Dict[str, Any]] = []
        for field, candidate in best_wdc_pave_candidates_by_field(info, stats).items():
            threshold = thresholds.get(field, 0.5)
            if float(candidate["score"]) < threshold:
                continue
            jsonld[field] = candidate["value"]
            admitted_trace.append(
                {
                    "path": field,
                    "value": candidate["value"],
                    "source": candidate["source"],
                    "score": round(float(candidate["score"]), 6),
                    "threshold": threshold,
                    "label": candidate["label"],
                    "agree": candidate["agree"],
                    "shape": candidate["shape"],
                }
            )
        predictions[ex_id] = {
            "jsonld": jsonld,
            "admission_learner": {
                "enabled": True,
                "train_split": train_split,
                "admitted": admitted_trace,
            },
        }
        trace[ex_id] = admitted_trace
    return predictions, trace


def is_valid_jsonld(pred: Dict[str, Any], schema_type: str, schema_index: Dict[str, Any]) -> bool:
    try:
        json.dumps(pred)
    except TypeError:
        return False
    if "@type" not in pred:
        return False
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    for key in pred:
        root = key.split(":")[-1]
        if root not in allowed:
            return False
    return True


def split_examples(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    examples = sorted(examples, key=lambda x: (x["schema_type"], x["source"], x["id"]))
    train, test = [], []
    by_type_source: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ex in examples:
        key = (ex["schema_type"], ex.get("source", "unknown"))
        by_type_source.setdefault(key, []).append(ex)
    for items in by_type_source.values():
        if len(items) == 1:
            test.extend(items)
            continue
        cut = max(1, int(len(items) * 0.6))
        train.extend(items[:cut])
        test.extend(items[cut:])
    return {"train": train, "test": test}


def parse_wdc_nquads(path: Path, schema_type: str, limit_subjects: int = 250) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    by_subject: Dict[str, Dict[str, List[str]]] = {}
    line_re = re.compile(r'^\s*<([^>]+)>\s+<([^>]+)>\s+(.+?)\s+<[^>]+>\s*\.\s*$')
    literal_re = re.compile(r'^"((?:[^"\\]|\\.)*)"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?$')
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = line_re.match(line)
            if not match:
                continue
            subject, predicate, raw_obj = match.groups()
            prop = predicate.rstrip("/#").split("/")[-1].split("#")[-1]
            value = ""
            literal = literal_re.match(raw_obj.strip())
            if literal:
                value = bytes(literal.group(1), "utf-8").decode("unicode_escape", errors="ignore")
            elif raw_obj.strip().startswith("<"):
                value = raw_obj.strip()[1:].split(">")[0]
            if not value:
                continue
            record = by_subject.setdefault(subject, {})
            record.setdefault(prop, []).append(value)
            if len(by_subject) >= limit_subjects and subject not in by_subject:
                break
    examples = []
    for idx, (subject, props) in enumerate(by_subject.items(), 1):
        gold: Dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type, "url": subject}
        text_values = []
        for prop, values in props.items():
            clean_values = []
            for value in values:
                value = re.sub(r"\s+", " ", value).strip()
                if value and value not in clean_values:
                    clean_values.append(value)
            if not clean_values:
                continue
            gold[prop] = clean_values[0] if len(clean_values) == 1 else clean_values[:5]
            if prop not in {"url", "image"}:
                text_values.extend(clean_values[:3])
        if len(gold) <= 3 or not text_values:
            continue
        examples.append(
            {
                "id": f"wdc_{schema_type.lower()}_{idx:04d}",
                "schema_type": schema_type,
                "source": "Web Data Commons class-specific sample",
                "source_url": str(path.name),
                "text": "\n".join(text_values[:40]),
                "gold": gold,
            }
        )
    return examples


def parse_swde_dataset(root: Path, limit_per_site: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse SWDE pages and labels into schema.org JSON-LD examples."""

    root = Path(root)
    if not root.exists():
        return []
    limit_per_site = _swde_limit_per_site(limit_per_site)
    labels: Dict[Tuple[str, str], Dict[str, Dict[str, List[str]]]] = {}
    for gt_path in find_swde_groundtruth_files(root):
        vertical, site, attr = parse_swde_label_name(gt_path)
        if not vertical or not site or not attr:
            continue
        for page_id, values in parse_swde_groundtruth_file(gt_path).items():
            if not values:
                continue
            labels.setdefault((vertical, site), {}).setdefault(page_id, {})[attr] = values

    examples: List[Dict[str, Any]] = []
    for (vertical, site), page_labels in sorted(labels.items()):
        schema_type = SWDE_VERTICAL_SCHEMA_TYPES.get(vertical)
        if not schema_type:
            continue
        page_dirs = find_swde_page_dirs(root, vertical, site)
        if not page_dirs:
            continue
        kept = 0
        for page_id, attrs in sorted(page_labels.items()):
            if limit_per_site is not None and kept >= limit_per_site:
                break
            page_path = find_swde_page(page_dirs, page_id)
            if page_path is None:
                continue
            gold = build_swde_gold(vertical, attrs)
            if len(flatten_fields(gold)) <= 2:
                continue
            page = read_text(page_path)
            text = compact_swde_text(strip_tags(page))
            source_url = extract_base_href(page) or page_path.name
            examples.append(
                {
                    "id": f"swde_{vertical}_{site}_{page_id}",
                    "schema_type": schema_type,
                    "source": f"SWDE {vertical}/{site}",
                    "source_url": source_url,
                    "raw_html_path": str(page_path),
                    "site": site,
                    "vertical": vertical,
                    "text": text,
                    "gold": gold,
                }
            )
            kept += 1
    return examples


def _swde_limit_per_site(limit_per_site: Optional[int]) -> Optional[int]:
    if limit_per_site is not None:
        return limit_per_site
    raw = os.environ.get("SWDE_MAX_PER_SITE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def find_swde_groundtruth_files(root: Path) -> List[Path]:
    roots = [root / "groundtruth", root]
    paths: List[Path] = []
    for base in roots:
        if base.exists():
            paths.extend(base.glob("*/*.txt"))
            paths.extend(base.glob("*.txt"))
    seen = set()
    unique = []
    for path in sorted(paths):
        key = str(path.resolve())
        if key not in seen and re.match(r"^[a-z]+-[^-]+-.+\.txt$", path.name):
            unique.append(path)
            seen.add(key)
    return unique


def parse_swde_label_name(path: Path) -> Tuple[str, str, str]:
    parts = path.stem.split("-", 2)
    if len(parts) == 3:
        return parts[0].lower(), parts[1].lower(), parts[2].lower()
    return "", "", ""


def parse_swde_groundtruth_file(path: Path) -> Dict[str, List[str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not lines:
        return {}
    start = 0
    header = lines[0].split("\t")
    if len(header) == 3 and header[0].lower() in SWDE_VERTICAL_SCHEMA_TYPES:
        start = 2
    out: Dict[str, List[str]] = {}
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        page_id = parts[0].strip()
        if len(parts) > 2 and parts[1].strip().isdigit():
            count = int(parts[1])
            values = parts[2 : 2 + count]
        else:
            values = parts[1:]
        cleaned = [clean_swde_value(value) for value in values]
        cleaned = [value for value in cleaned if value]
        if cleaned:
            out[page_id] = cleaned
    return out


def clean_swde_value(value: str) -> str:
    value = html.unescape(value).replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value).strip()
    if not value or value == "<NULL>":
        return ""
    return value


def find_swde_page_dirs(root: Path, vertical: str, site: str) -> List[Path]:
    vertical_dir = root / vertical
    if not vertical_dir.exists():
        return []
    patterns = [f"{vertical}-{site}*", site, f"{site}*"]
    dirs: List[Path] = []
    for pattern in patterns:
        dirs.extend(path for path in vertical_dir.glob(pattern) if path.is_dir())
    seen = set()
    unique = []
    for path in sorted(dirs):
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def find_swde_page(page_dirs: List[Path], page_id: str) -> Optional[Path]:
    for directory in page_dirs:
        for suffix in (".htm", ".html"):
            path = directory / f"{page_id}{suffix}"
            if path.exists():
                return path
    return None


def build_swde_gold(vertical: str, attrs: Dict[str, List[str]]) -> Dict[str, Any]:
    schema_type = SWDE_VERTICAL_SCHEMA_TYPES[vertical]
    gold: Dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type}
    attr_map = SWDE_ATTRIBUTE_MAP.get(vertical, {})
    for attr, values in sorted(attrs.items()):
        target = attr_map.get(attr)
        if not target:
            continue
        value: Any = values[0] if len(values) == 1 else values[:8]
        set_jsonld_path(gold, target, value)
    return gold


def set_jsonld_path(obj: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = obj
    for part in parts[:-1]:
        default_child: Dict[str, Any]
        if part == "offers":
            default_child = {"@type": "Offer"}
        elif part == "aggregateRating":
            default_child = {"@type": "AggregateRating"}
        else:
            default_child = {}
        if not isinstance(cur.get(part), dict):
            cur[part] = default_child
        elif part in {"offers", "aggregateRating"} and "@type" not in cur[part]:
            cur[part]["@type"] = default_child["@type"]
        cur = cur[part]
    cur[parts[-1]] = value


def has_jsonld_path(obj: Dict[str, Any], dotted_path: str) -> bool:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def compact_swde_text(text: str) -> str:
    max_chars = int(os.environ.get("SWDE_TEXT_CHAR_LIMIT", "30000") or "30000")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def extract_base_href(page: str) -> str:
    match = re.search(r'(?is)<base\s+href=["\']([^"\']+)["\']', page)
    return html.unescape(match.group(1)).strip() if match else ""


def extract_warc_html(raw_gzip: bytes) -> str:
    data = gzip.decompress(raw_gzip)
    # WARC headers, HTTP headers, then payload. Split twice on blank lines.
    parts = re.split(rb"\r?\n\r?\n", data, maxsplit=2)
    if len(parts) == 3:
        payload = parts[2]
    elif len(parts) == 2:
        payload = parts[1]
    else:
        payload = data
    return payload.decode("utf-8", errors="replace")


def parse_books_toscrape_html(
    path: Path,
    idx: int,
    source_name: str = "Common Crawl CC-MAIN-2023-50 books.toscrape capture",
    id_prefix: str = "cc_books_product",
    source_url: str | None = None,
) -> Dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    page = read_text(path)
    title = first_match(page, r"(?is)<h1[^>]*>(.*?)</h1>")
    if not title:
        return None
    description = ""
    desc_match = re.search(
        r"(?is)<div id=\"product_description\".*?</div>\s*<p>(.*?)</p>",
        page,
    )
    if desc_match:
        description = strip_tags(desc_match.group(1))
    table = {}
    for row in re.findall(r"(?is)<tr>\s*<th>(.*?)</th>\s*<td>(.*?)</td>\s*</tr>", page):
        key = normalize_text(strip_tags(row[0])).replace(" ", "_")
        value = strip_tags(row[1])
        table[key] = value
    price = table.get("price_excl_tax") or table.get("price_incl_tax") or first_match(page, r"£\d+(?:\.\d{2})?")
    availability = table.get("availability", "")
    review_count = table.get("number_of_reviews", "")
    upc = table.get("upc", "")
    if not upc and not table.get("price_excl_tax"):
        return None
    rating = ""
    rating_match = re.search(r'(?is)<p class="star-rating\s+([A-Za-z]+)"', page)
    if rating_match:
        rating = rating_match.group(1)
    rating_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    rating_value = rating_map.get(rating.lower(), "")
    text = strip_tags(page)
    # The books.toscrape rating is encoded as a DOM class rather than visible text.
    # Preserve it as evidence so extraction does not need to read the gold record.
    if rating_value:
        text = f"{text}\nStar rating: {rating_value} stars"
    gold: Dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": strip_tags(title),
    }
    if description:
        gold["description"] = description
    if upc:
        gold["sku"] = upc
    if price or availability:
        gold["offers"] = {"@type": "Offer"}
        if price:
            gold["offers"]["price"] = price
            gold["offers"]["priceCurrency"] = "GBP"
        if availability:
            gold["offers"]["availability"] = availability
    if rating_value or review_count:
        gold["aggregateRating"] = {"@type": "AggregateRating"}
        if rating_value:
            gold["aggregateRating"]["ratingValue"] = rating_value
        if review_count:
            gold["aggregateRating"]["reviewCount"] = review_count
    return {
        "id": f"{id_prefix}_{idx:04d}",
        "schema_type": "Product",
        "source": source_name,
        "source_url": source_url or path.name,
        "text": text,
        "gold": gold,
    }


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return ""
    return strip_tags(match.group(1) if match.groups() else match.group(0))


def metric_rows(metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    rows = []
    for method, vals in metrics.items():
        rows.append(
            {
                "method": method,
                "examples": str(int(vals["examples"])),
                "precision": f"{vals['precision']:.3f}",
                "recall": f"{vals['recall']:.3f}",
                "f1": f"{vals['f1']:.3f}",
                "site_macro_f1": f"{vals.get('site_macro_f1', 0.0):.3f}",
                "type_macro_f1": f"{vals.get('type_macro_f1', 0.0):.3f}",
                "jsonld_validity": f"{vals['jsonld_validity']:.3f}",
                "hallucination_rate": f"{vals['hallucination_rate']:.3f}",
                "unsupported_fields": str(int(vals.get("unsupported_fields", 0))),
                "predicted_fields": str(int(vals.get("predicted_fields", 0))),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(row[field] for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_table(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_bar_chart(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["f1", "jsonld_validity", "hallucination_rate"]
    width, height = 760, 420
    margin = 70
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    colors = {
        "baseline_regex": "#4C78A8",
        "schema_only": "#54A24B",
        "evidence_only": "#B279A2",
        "no_type_aware_aliases": "#72B7B2",
        "no_nested_mapping": "#E45756",
        "no_schema_validator": "#9D755D",
        "no_evidence_validator": "#FF9DA6",
        "schemarag": "#F58518",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">SchemaRAG vs Baseline</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    group_w = chart_w / len(metrics)
    bar_w = min(52, max(14, (group_w - 70) / max(1, len(rows))))
    gap = min(12, max(4, bar_w * 0.35))
    for i, metric in enumerate(metrics):
        total_bar_w = len(rows) * bar_w + max(0, len(rows) - 1) * gap
        x0 = margin + i * group_w + (group_w - total_bar_w) / 2
        svg.append(f'<text x="{margin + i * group_w + group_w / 2}" y="{height-margin+28}" text-anchor="middle" font-family="Arial" font-size="13">{metric}</text>')
        for j, row in enumerate(rows):
            value = float(row[metric])
            bar_h = value * chart_h
            x = x0 + j * (bar_w + gap)
            y = height - margin - bar_h
            color = colors.get(row["method"], "#999")
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            svg.append(f'<text x="{x+bar_w/2}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
    legend_x = width - 230
    for idx, row in enumerate(rows):
        y = 55 + idx * 24
        svg.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" fill="{colors.get(row["method"], "#999")}"/>')
        svg.append(f'<text x="{legend_x+22}" y="{y}" font-family="Arial" font-size="13">{row["method"]}</text>')
    for tick in range(0, 6):
        value = tick / 5
        y = height - margin - value * chart_h
        svg.append(f'<line x1="{margin-5}" y1="{y:.1f}" x2="{margin}" y2="{y:.1f}" stroke="#333"/>')
        svg.append(f'<text x="{margin-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def now_utcish() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

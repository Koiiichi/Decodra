"""Benchmark schema suite for Decodra."""

from __future__ import annotations

from pydantic import BaseModel


class PersonSchema(BaseModel):
    """Simple person extraction schema."""

    name: str
    age: int
    email: str


class ProductSchema(BaseModel):
    """Simple product extraction schema."""

    name: str
    price: float
    in_stock: bool


class LocationSchema(BaseModel):
    """Simple location extraction schema."""

    city: str
    country: str
    timezone: str


class CompanyAddressSchema(BaseModel):
    """Nested company address schema."""

    street: str
    city: str
    country: str


class CompanySchema(BaseModel):
    """Nested company extraction schema."""

    name: str
    founded: int
    address: CompanyAddressSchema
    employee_count: int


class VenueSchema(BaseModel):
    """Nested event venue schema."""

    name: str
    city: str


class EventSchema(BaseModel):
    """Nested event extraction schema."""

    title: str
    date: str
    venue: VenueSchema
    capacity: int


class AuthorSchema(BaseModel):
    """Nested article author schema."""

    name: str
    affiliation: str


class ArticleSchema(BaseModel):
    """Nested article extraction schema."""

    title: str
    author: AuthorSchema
    published: str
    word_count: int


class MetricsSchema(BaseModel):
    """Typed model evaluation metrics schema."""

    accuracy: float
    precision: float
    recall: float
    f1: float


class RepositorySchema(BaseModel):
    """Typed software repository schema."""

    name: str
    stars: int
    language: str
    license: str
    is_public: bool


class InvoiceSchema(BaseModel):
    """Typed invoice extraction schema."""

    invoice_id: str
    amount: float
    currency: str
    paid: bool
    items_count: int


class ResearchPaperSchema(BaseModel):
    """Complex research paper schema."""

    title: str
    authors: list[str]
    year: int
    venue: str
    citation_count: int
    open_access: bool


class JobPostingSchema(BaseModel):
    """Complex job posting schema."""

    title: str
    company: str
    location: str
    salary_min: int
    salary_max: int
    remote: bool
    experience_years: int


class MedicalRecordSchema(BaseModel):
    """Complex medical record schema."""

    patient_id: str
    age: int
    diagnosis: str
    severity: str
    follow_up_required: bool


BENCHMARK_SCHEMAS: list[type[BaseModel]] = [
    PersonSchema,
    ProductSchema,
    LocationSchema,
    CompanySchema,
    EventSchema,
    ArticleSchema,
    MetricsSchema,
    RepositorySchema,
    InvoiceSchema,
    ResearchPaperSchema,
    JobPostingSchema,
    MedicalRecordSchema,
]

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    schema.__name__: schema for schema in BENCHMARK_SCHEMAS
}

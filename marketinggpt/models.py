from typing import List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import date
from pydantic import BaseModel, validator
from datetime import datetime


class Customer(BaseModel):
    customer_id: input
    name: str
    email: EmailStr
    phone_number: int
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    income: Optional[int] = None
    age: int
    interests: Optional[str] = None
    purchase_history: Optional[str] = None



class Product(BaseModel):
    product_id: int
    product_name: str
    category: str
    price: int
    discount: Optional[int] = None
    promotion_start_date: Optional[date] = None
    promotion_end_date: Optional[date] = None
    promotion_details: Optional[str] = None



class SearchQuery(BaseModel):
    query: str


class SearchResult(BaseModel):
    customers: Optional[List[Customer]] = None
    products: Optional[List[Product]] = None
    total_customers: Optional[int] = None
    total_products: Optional[int] = None
    
    
class Searcher(BaseModel):
    query: SearchQuery
    results: Optional[SearchResult] = None
    

class Email(BaseModel):
    email_id: int
    customer: Customer
    subject: str
    content: str
    products: List[Product]
    promotions: List[Promotion]
    sent_at: Optional[datetime] = None


class SearchState(BaseModel):
    query: str = Field(default="")
    tool_requests: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_result: Dict[str, Any] = Field(default_factory=dict)
    user_confirmations: List[bool] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    
    


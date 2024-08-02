# 📧 MarketingGPT

## 📋 Introduction

MarketingGPT is an automated email marketing system designed to send personalized marketing emails to specific customer groups. The system uses agents to search for customer and product information, then creates and sends appropriate marketing emails.

## 📊 Data Fields

### 📇 Customer Data

- **🆔 customer_id**: Customer ID
- **👤 name**: Customer Name
- **📧 email**: Customer Email
- **📞 phone_number**: Phone Number
- **♂️ gender**: Gender
- **💍 marital_status**: Marital Status
- **💰 income**: Income
- **🎂 age**: Age
- **🎯 interests**: Interests
- **🛒 purchase_history**: Purchase History

### 🛍️ Product Data

- **🆔 product_id**: Product ID
- **📦 product_name**: Product Name
- **📂 category**: Product Category
- **💵 price**: Product Price
- **🔖 discount**: Discount Rate
- **📅 promotion_start_date**: Promotion Start Date
- **📅 promotion_end_date**: Promotion End Date
- **📝 promotion_details**: Promotion Details

## 🔄 System Workflow

```plaintext
+---------------+        +-----------+        +-------------+
| User Request  | -----> | Searcher  | -----> | Writer      |
+---------------+        +-----------+        +-------------+
                                        |           |
                                        |           |
                                 +-------------+    |
                                 | Email Sender| ---+
                                 +-------------+
📥 User Request: The user inputs a request for an email marketing campaign (e.g., send a promotion email to single customers).

🔍 Searcher: This agent searches for and retrieves customer and product information from the database.

✍️ Writer: This agent creates marketing email content based on information from the Searcher.

📤 Email Sender: The system sends marketing emails to the selected customer list.

🕵️‍♂️ Agent Descriptions

### 🔍 Searcher Agent

#### Functions

- Search for customer information based on specific criteria (e.g., marital status, interests, purchase history).
- Retrieve product information and promotions from the database.

### ✍️ Writer Agent

#### Functions

- Create personalized marketing email content based on customer and product information.

### 📤 Email Sender Agent

#### Functions

- Send marketing emails to the selected customer list.
- Track email sending status and generate reports on campaign results.

"# MarketingGPT_Langgraph" 
"# MarketingGPT_Langgraph" 

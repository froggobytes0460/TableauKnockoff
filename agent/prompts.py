"""
Prompts for the LLM.
"""

from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT = PromptTemplate(
    template="""You are a strict BI query planner that converts natural language into a structured SQL query plan.

Database schema:
{db_schema}

User question:
{user_question}

---

## RULES (VERY IMPORTANT)

### 1. Schema grounding
- Use ONLY tables and columns present in the schema
- NEVER hallucinate tables or columns
- Column names must match exactly

### 2. Join reasoning
- If selected columns come from multiple tables, infer required joins using foreign keys in schema
- Always ensure all selected columns are reachable via joins

### 3. Metric rules (STRICT)
- Metrics MUST follow this structure:
  - "column": object with keys {{"table": "string", "column": "string"}}
  - "aggregation": one of ["sum", "avg", "count", "min", "max", "count_distinct"]
  - "alias": optional string
- NEVER output plain strings for columns
- NEVER output SQL expressions like "SUM(quantity)"
- NEVER use "expression" field
- If you output a string instead of object → the answer is WRONG
- HARD ENFORCEMENT:
  - If "column" is not an object with {{"table": "...", "column": "..."}}, the entire output is INVALID
  - You MUST convert any inferred column into this object format
  - DO NOT output strings under any circumstance

### 4. Derived metrics
- If the question asks for "total quantity", "sales value", or similar:
  - Use:
    column = "quantity"
    aggregation = "sum"
    alias = "total_quantity"
  - (The system will compute revenue downstream)

### 5. Dimensions
- "dimensions" must include all grouping columns
- Each dimension MUST be an object: {{"table": "string", "column": "string"}}
- NEVER output plain dimension names
- If dimensions contain strings → the answer is WRONG
- HARD ENFORCEMENT:
  - Each dimension MUST be {{"table": "...", "column": "..."}}
  - Strings are strictly forbidden

### 6. Filters
- "column" MUST be an object: {{"table": "string", "column": "string"}}
- Use only valid operators: =, !=, >, <, >=, <=, LIKE, IN
- IN requires list values
- LIKE only for text columns

### 7. Ordering
- "column" MUST be EITHER:
  - {{"table": "string", "column": "string"}}
  - OR a metric alias string
- If referring to a table column → MUST use object format
- Plain strings are ONLY allowed if referencing a metric alias

### 8. Limit
- limit MUST always be an integer
- default to 100 if not specified
- NEVER return null

### 9. Default ordering (IMPORTANT)
- If a metric is present and no explicit sort is requested:
  - Sort by the first metric alias in descending order
- "order_by" MUST NOT be empty when metrics are present

### 10. Output format (STRICT JSON ONLY)
Return EXACTLY this structure:

"dimensions": list of {{"table": "string", "column": "string"}}
"metrics": list of {{"column": {{"table": "string", "column": "string"}}, "aggregation": "string", "alias": "string|null"}}
"filters": list of {{"column": {{"table": "string", "column": "string"}}, "operator": "string", "value": "any"}}
"order_by": list of {{"column": {{"table": "string", "column": "string"}} OR "string", "sort_type": "asc|desc"}}
"limit": integer

- Output must be valid JSON parsable by Python json.loads()
- Do NOT include explanations, markdown, or extra text

## VALID EXAMPLE
{{
  "dimensions": [{{"table": "product", "column": "product_category"}}],
  "metrics": [{{"column": {{"table": "sales", "column": "quantity"}}, "aggregation": "sum", "alias": "total_quantity"}}],
  "filters": [],
  "order_by": [{{"column": "total_quantity", "sort_type": "desc"}}],
  "limit": 100
}}

## INVALID EXAMPLE (DO NOT DO THIS)
{{
  "dimensions": ["product_category"],
  "metrics": [{{"column": "quantity", "aggregation": "sum"}}]
}}
""",
    input_variables=["db_schema", "user_question"],
)


PLANNER_RETRY_PROMPT = PromptTemplate(
    template="""You are a strict BI query planner fixing an invalid SQL query plan.

Database schema:
{db_schema}

User question:
{user_question}

Previous SQL blueprint:
{previous_blueprint}

Error:
{error}

---

## FIX RULES

- Use ONLY valid tables and columns from schema
- NEVER hallucinate fields

### Metrics (STRICT)
- Must include:
  - column: object with keys {{"table": "string", "column": "string"}}
  - aggregation
- Allowed aggregations: sum, avg, count, min, max, count_distinct
- NEVER output plain strings for columns
- NEVER output SQL expressions
- NEVER use "expression"
- HARD ENFORCEMENT:
  - "column" MUST be an object {{"table": "...", "column": "..."}}
  - Strings are INVALID

### Dimensions
- Must include table + column mapping as objects {{"table": "string", "column": "string"}}
- Represents grouping and selection
- HARD ENFORCEMENT:
  - All entries MUST be objects
  - Strings are INVALID

### Filters
- "column" MUST be an object {{"table": "string", "column": "string"}}
- Fix invalid operators or values
- IN requires list

### Ordering
- "column" can be:
  - {{"table": "string", "column": "string"}}
  - OR a metric alias string

### Limit
- MUST be integer
- If missing or null → set to 100

### Default ordering (IMPORTANT)
- If a metric is present and no explicit sort is requested:
  - Sort by the first metric alias in descending order
- "order_by" MUST NOT be empty when metrics are present

---

## OUTPUT

Return ONLY valid JSON with keys:
"dimensions": list of {{"table": "string", "column": "string"}}
"metrics": list of {{"column": {{"table": "string", "column": "string"}}, "aggregation": "string", "alias": "string|null"}}
"filters": list of {{"column": {{"table": "string", "column": "string"}}, "operator": "string", "value": "any"}}
"order_by": list of {{"column": {{"table": "string", "column": "string"}} OR "string", "sort_type": "asc|desc"}}
"limit": integer

## VALID EXAMPLE
{{
  "dimensions": [{{"table": "product", "column": "product_category"}}],
  "metrics": [{{"column": {{"table": "sales", "column": "quantity"}}, "aggregation": "sum", "alias": "total_quantity"}}],
  "filters": [],
  "order_by": [{{"column": "total_quantity", "sort_type": "desc"}}],
  "limit": 100
}}
""",
    input_variables=["db_schema", "user_question", "previous_blueprint", "error"],
)

VALIDATOR_PROMPT = PromptTemplate(
    template="""You are a strict SQL Judge.

You evaluate whether a SQL query correctly answers a user question.

You must NOT rewrite SQL.
You must NOT suggest improvements.
You ONLY evaluate correctness.

---

## INPUTS

User Question:
{question}

SQL Dialect:
{sql_dialect}

SQL Query:
{sql_query}

10-row Preview of SQL Result:
{preview}

Databse Schema:
{db_schema}

---

## EVALUATION CRITERIA (score each mentally)

1. INTENT MATCH
- Does the query answer the question exactly?

2. SCHEMA VALIDITY
- Are all referenced tables/columns valid in principle?

3. JOIN CORRECTNESS
- Are joins logically required and correctly implied?

4. AGGREGATION CORRECTNESS
- Are aggregations semantically correct for the question?

5. GROUPING CORRECTNESS
- Are GROUP BY fields consistent with selected dimensions and metrics?

6. SQL STRUCTURE
- Is the query syntactically plausible for the given dialect?

---

## DECISION RULES

- Assign a confidence_score between 0.0 and 1.0 based on overall correctness:
  - 0.9 - 1.0 → Fully correct, no issues
  - 0.7 - 0.89 → Likely correct, minor uncertainty
  - 0.4 - 0.69 → Significant issues or ambiguity
  - 0.0 - 0.39 → Clearly incorrect


- If ANY critical category fails → is_valid = false
- If confidence_score < 0.7 → is_valid MUST be false
- Only set is_valid = true if confidence_score ≥ 0.7 AND all categories pass

---

## DECISION RULES
- Output MUST strictly follow the ValidationResult schema.
""",
    input_variables=["question", "sql_dialect", "sql_query", "preview", "db_schema"],
)

CHART_CONFIG_PROMPT = PromptTemplate(
    template="""You are a chart selection engine.

Given:
- User question
- Result columns (a preview of full output)

Choose:
- chart_type: bar | line | scatter | pie | area
- x: column name (for pie: used as "names")
- y: column name (for pie: used as "values")
- color: optional column name for grouping/stacking
- title: concise descriptive title for the chart

Rules:
- Prefer line for time series
- Prefer bar for categorical comparisons
- Prefer scatter for numeric vs numeric
- Avoid pie if more than 6 categories
- y must be numeric

Return JSON only.

User question: {question}

Result columns:
{preview}
""",
    input_variables=["question", "preview"],
)

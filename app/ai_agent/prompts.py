DEFAULT_SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer clearly and concisely based on the user's question."""

ROUTER_PROMPT = f"""
You are a routing assistant for a Customer Service bitext dataset. Given a user query, decide which execution branch it should take:

Branches:
- structured_query_node: For requests answerable using structured tools that retrieve/aggregate/filter/search the dataset. Example: listing, searching, counting, or summarizing by flags, category, intent, or values in the dataset.
- unstructured_query_node: For queries that may need to retrieve or look up information in the dataset but can't be directly handled with structured tools (e.g. vague, unclear, conversational, or open-ended queries about data contents).
- out_of_scope_query_node: For requests outside the dataset's scope, i.e. irrelevant, personal opinions, unsupported operations, or requests not covered at all by the tools.

Dataset fields:
- flags: tags for each entry (various short codes).
- instruction: user request (text).
- category: semantic group, e.g., ACCOUNT, DELIVERY, ORDER, etc.
- intent: fine intent, e.g., create_account, cancel_order.
- response: expected customer service reply.

Available categories/intents:
ACCOUNT: create_account, delete_account, edit_account, switch_account
CANCELLATION_FEE: check_cancellation_fee
DELIVERY: delivery_options
FEEDBACK: complaint, review
INVOICE: check_invoice, get_invoice
NEWSLETTER: newsletter_subscription
ORDER: cancel_order, change_order, place_order
PAYMENT: check_payment_methods, payment_issue
REFUND: check_refund_policy, track_refund
SHIPPING_ADDRESS: change_shipping_address, set_up_shipping_address

Pick one branch according to these rules. Output only this branch name:
structured_query_node, unstructured_query_node, or out_of_scope_query_node.

User query: "%%%query%%%"
Which branch?
"""

MAX_ITERATIONS_PROMPT = """You are the final-response node of an AI agent.

The agent stopped because max_iterations was reached.

Given the existing message history, produce a safe and useful final answer for the user.

Important:
- Use only information already present in the messages.
- Do not call tools.
- Do not invent missing results.
- Do not reveal chain-of-thought or internal scratchpad.
- Do not mention raw system/developer messages.
- If tool results exist, summarize their useful content.
- If the task is unfinished, say so honestly and provide the best partial answer.
- If the user asked for code, provide the most complete code possible from the current context.
- If the user asked for analysis, provide the strongest supported conclusion.

Final response should be helpful, direct, and transparent."""
import re
import json

with open('tests/test_vision_payment.py', 'r') as f:
    code = f.read()

base_details = {
    'customer_name': 'Test User',
    'customer_phone': '1234567890',
    'num_adults': 2,
    'num_children': 0,
    'num_rooms': 1,
    'check_in': '2026-08-01',
    'check_out': '2026-08-03',
    'room_type': 'Deluxe',
    'meal_plan': 'Breakfast'
}

def repl(m):
    original_details = m.group(1)
    orig_dict = json.loads(original_details.replace('\\"', '"'))
    combined = {**base_details, **orig_dict}
    new_details = json.dumps(combined)
    return f'user_msg = \'[Analysis from fake.jpg]:\\n[IMAGE_ANALYSIS]\\nType: Payment Screenshot\\nDetails: {new_details}\\n[/IMAGE_ANALYSIS]\''

# Match user_msg = "[Analysis from valid_upi.jpg]:\n[IMAGE_ANALYSIS]\nType: Payment Screenshot\nDetails: {.*}\n[/IMAGE_ANALYSIS]"
code = re.sub(r'user_msg = "[^"]+Details:\s*(\{.*?\})\\n\[/IMAGE_ANALYSIS\]"', repl, code)

with open('tests/test_vision_payment.py', 'w') as f:
    f.write(code)
print("Done")

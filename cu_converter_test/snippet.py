from cu_time_converter_stable_v2_1_6 import gregorian_to_cu, cu_to_gregorian, calculate_cu_duration, run_tests

print("🌟 Test 1: Running Full Test Suite 🌟")
run_tests()

print("\n🌟 Test 2: Gregorian to CU-Time (02/29/2000) 🌟")
print(gregorian_to_cu("02/29/2000 00:00:00 UTC"))

print("\n🌟 Test 3: CU-Time to Gregorian (3094134044923.672659) 🌟")
print(cu_to_gregorian("3094134044923.672659"))

print("\n🌟 Test 4: Duration between 02/29/2000 and 05/23/2025 🌟")
print(calculate_cu_duration("02/29/2000 00:00:00 UTC", "05/23/2025 19:59:00 UTC"))

print("\n🌟 Test 5: Gregorian to CU-Time (28000000000 BCE) 🌟")
print(gregorian_to_cu("01/01/28000000000 BCE 00:00:00 UTC"))
def dialpad(read_keypad):
    code = ""

    while True:
        key = read_keypad()   # Must return ONE key press

        if not key:
            continue

        # DIGITS 0–9
        if key in "1234567890":
            code += key
            code = code[-3:]  # keep last 3 digits only

        # BACKSPACE
        elif key == "*":
            code = code[:-1]

        # SUBMIT
        elif key == "#":
            if len(code) == 3:
                return code
            else:
                code = ""  # reset if invalid

        # Ignore everything else
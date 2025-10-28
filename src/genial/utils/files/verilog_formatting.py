def unwrap_lines(design_lines: list[str], eol_char=";") -> list[str]:
    """
    This function unwraps a line-wrapped file expecting each line to end up with the end of line character specified in input argument.
    """
    new_lines = []
    buffer = ""
    for line in design_lines:
        if eol_char in line:
            buffer += line + "\n"
            new_lines.append(buffer)
            buffer = ""
        else:
            buffer += line

    if buffer:
        new_lines.append(buffer + "\n")

    return new_lines


def reformat_code(input_str):
    def find_matching_paren(s, start):
        # Finds the index of the matching closing parenthesis for the opening at index 'start'
        stack = []
        for i in range(start, len(s)):
            if s[i] == "(":
                stack.append(i)
            elif s[i] == ")":
                stack.pop()
                if not stack:
                    return i
        return -1  # No matching closing parenthesis found

    def split_on_commas(s):
        # Splits the string on commas that are not inside parentheses
        result = []
        current = ""
        depth = 0
        for c in s:
            if c == "," and depth == 0:
                result.append(current)
                current = ""
            else:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                current += c
        result.append(current)
        return result

    def process_content(s):
        # Processes the content inside the parentheses
        items = split_on_commas(s)
        items = [item.strip() for item in items]
        formatted_items = []
        for idx, item in enumerate(items):
            # Add commas except for the last item
            comma = "," if idx < len(items) - 1 else ""
            formatted_items.append(f" {item}{comma}")
        return "\n".join(formatted_items)

    result = ""
    i = 0
    while i < len(input_str):
        if input_str[i] == "(":
            # Find the matching closing parenthesis
            end = find_matching_paren(input_str, i)
            if end == -1:
                # No matching closing parenthesis; append the rest and break
                result += input_str[i:]
                break
            else:
                # Process the content inside the parentheses
                content = input_str[i + 1 : end]
                formatted_content = process_content(content)

                # Collect any characters after the closing parenthesis (like semicolons)
                after_paren = ""
                j = end + 1
                while j < len(input_str) and input_str[j] in " \t;":
                    after_paren += input_str[j]
                    j += 1
                i = j - 1  # Adjust the index to continue from the correct position

                # Append the formatted content
                result += "(\n" + formatted_content + "\n)" + after_paren
        else:
            result += input_str[i]
        i += 1
    return result

import re 

def extract_fix_code(completion):
        pattern = re.compile(r"```java\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(completion)
        print(matches)
        if len(matches) > 0:
            return matches[0].strip()
        else:
            return None
        
string = '```java\nreturn stringBuilder.toString().replaceAll("\\\\s+", "").trim();\n```'
print(extract_fix_code(string))

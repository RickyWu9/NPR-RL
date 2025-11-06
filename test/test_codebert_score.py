import code_bert_score
model_path = "/root/wy/NPR-RL/assets/model/codebert_base"

prediction = "public static String suggest(String words) {\
String[] wordArray = words.split(\",\");\
ArrayList<String> suggestedBrainKey = new ArrayList<String>();\
assert (wordArray.length == DICT_WORD_COUNT);\
SecureRandom secureRandom = SecureRandomGenerator.getSecureRandom();\
int index;\
for (int i = 0; i < BRAINKEY_WORD_COUNT; i++) {\
index = secureRandom.nextInt(DICT_WORD_COUNT - 1);\
suggestedBrainKey.add(wordArray[index].toUpperCase());\
}\
StringBuilder stringBuilder = new StringBuilder();\
for(String word : suggestedBrainKey){\
stringBuilder.append(word);\
stringBuilder.append(\" \");\
}\
return stringBuilder.toString().replaceAll(\"\\\\s+\", \"\").trim();\
}"

reference = "public static String suggest(String words) {\
String[] wordArray = words.split(\",\");\
ArrayList<String> suggestedBrainKey = new ArrayList<String>();\
assert (wordArray.length == DICT_WORD_COUNT);\
SecureRandom secureRandom = SecureRandomGenerator.getSecureRandom();\
int index;\
for (int i = 0; i < BRAINKEY_WORD_COUNT; i++) {\
index = secureRandom.nextInt(DICT_WORD_COUNT - 1);\
suggestedBrainKey.add(wordArray[index].toUpperCase());\
}\
StringBuilder stringBuilder = new StringBuilder();\
for(String word : suggestedBrainKey){\
stringBuilder.append(word);\
stringBuilder.append(\" \");\
}\
return stringBuilder.toString().trim().toLowerCase();\
}"
print(prediction)
result = code_bert_score.score(cands=[prediction], refs=[reference], lang='java', model_type=model_path)
print(result)
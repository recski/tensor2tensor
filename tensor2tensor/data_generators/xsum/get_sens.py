import json
import sys

from xsum import preprocess


field_name = sys.argv[1]

for line in sys.stdin:
    d = json.loads(line)
    for sen in d[field_name]:
        sys.stdout.write(preprocess(sen).encode('utf-8') + " ")
    sys.stdout.write('\n')

import sys
paths=[
    r"C:\Users\Administrator\Downloads\lm-evaluation-harness\lm_eval\models\optimum_lm_genai.py",
    r"C:\Users\Administrator\Downloads\lm-evaluation-harness\lm_eval\models\optimum_lm_genai_117228b3.py",
]
any_changed=False
for p in paths:
    try:
        b=open(p,'rb').read()
    except Exception as e:
        print('ERROR reading',p,e)
        continue
    n=b.count(b'\x00')
    print(p,'nulls=',n)
    if n>0:
        b2=b.replace(b'\x00',b'')
        open(p,'wb').write(b2)
        print('Cleaned',p)
        any_changed=True
if not any_changed:
    print('No files changed')
else:
    print('Done cleaning')

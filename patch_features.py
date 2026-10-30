"""One-time patch: let features.py use an explicit Q&A marker if present."""
p = "src/features.py"
s = open(p).read()
if "QA_TOKEN" in s:
    print("Already patched.")
else:
    old_def = 'def split_sections(text):\n    """Split into (prepared remarks, Q&A). Q&A is \'\' if no marker is found."""\n    text = str(text)\n'
    new_def = ('QA_TOKEN = "[[QA_START]]"   # inserted by make_dataset.py where the Q&A begins\n\n\n'
               'def split_sections(text):\n    """Split into (prepared remarks, Q&A). Q&A is \'\' if no marker is found."""\n    text = str(text)\n'
               '    if QA_TOKEN in text:\n        prep, qa = text.split(QA_TOKEN, 1)\n        return prep, qa\n')
    assert old_def in s, "split_sections not found; file differs from expected"
    s = s.replace(old_def, new_def)
    old_all = '("all", str(text))'
    assert old_all in s, "all-text tuple not found"
    s = s.replace(old_all, '("all", str(text).replace(QA_TOKEN, " "))')
    open(p, "w").write(s)
    print("Patched src/features.py")

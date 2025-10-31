from pathlib import Path

p = Path(
    r"C:\Users\alexb\OneDrive - Université Libre de Bruxelles\MA2 EPFL\CS-433\test_imp\implementations.py"
)

# 1) Lire en 'utf-8-sig' -> supprime BOM si présent
text = p.read_text(encoding="utf-8-sig")

changed = False

# 2) Corriger le main guard s’il est sur une seule ligne
first_line = text.splitlines()[0] if text.splitlines() else ""
if first_line.strip().startswith(
    "if __name__=='__main__': print('OK')"
) or first_line.strip().startswith('if __name__ == "__main__": print("OK")'):
    rest = "\n".join(text.splitlines()[1:])
    if rest and not rest.endswith("\n"):
        rest += "\n"
    rest += 'if __name__ == "__main__":\n    print("OK")\n'
    text = rest
    changed = True

# 3) Sauvegarder sans BOM
p.write_text(text, encoding="utf-8")
print(
    "BOM retiré et main guard réécrit."
    if changed
    else "BOM retiré (pas de main guard à changer)."
)

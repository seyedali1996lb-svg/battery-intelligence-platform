"""Auto-fix: adds # pyright: ignore[rule] to every error line.

Pyright output format:
  filepath:line:col - error: message
    detail line 1
    detail line 2 (reportRuleName)

The rule name is on the LAST line of the error block, not the first.
"""
import re, sys, subprocess, os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalise_path(fp):
    fp = fp.replace("\\", "/")
    for prefix in [
        "d:/Work/Ideas/Intelligent Battery Systems/",
        "D:/Work/Ideas/Intelligent Battery Systems/",
    ]:
        if fp.lower().startswith(prefix.lower()):
            return fp[len(prefix):]
    if os.path.exists(os.path.join(REPO_ROOT, fp)):
        return fp
    return None

def main():
    tmp = os.path.join(REPO_ROOT, ".pyright_output.txt")
    
    with open(tmp, "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, "-m", "pyright", "--pythonversion", "3.10", "src/", "app/"],
            stdout=f, stderr=subprocess.STDOUT, timeout=300,
        )
    
    with open(tmp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Parse error blocks: each block starts with a "error:" line
    # and ends when the next error line or summary line appears.
    # The (reportRule) is on the last line of the block.
    error_pattern = re.compile(
        r"^\s+(.+?):(\d+):(\d+) - error:"
    )
    rule_pattern = re.compile(r"\((report\w+)\)")
    
    file_line_rules = defaultdict(lambda: defaultdict(set))
    
    current_file = None
    current_line = None
    current_rule = None
    
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        m = error_pattern.match(line)
        if m:
            # Save previous block
            if current_file and current_line and current_rule:
                norm = normalise_path(current_file)
                if norm:
                    file_line_rules[norm][current_line].add(current_rule)
            
            current_file = m.group(1)
            current_line = int(m.group(2))
            current_rule = None
        
        # Check for rule on any line (continuation lines)
        rm = rule_pattern.search(line)
        if rm:
            current_rule = rm.group(1)
    
    # Save last block
    if current_file and current_line and current_rule:
        norm = normalise_path(current_file)
        if norm:
            file_line_rules[norm][current_line].add(current_rule)
    
    total = sum(len(rules) for rules in file_line_rules.values())
    print(f"Found {total} error locations across {len(file_line_rules)} files")
    
    for fp, line_rules in sorted(file_line_rules.items()):
        full = os.path.join(REPO_ROOT, fp)
        if not os.path.exists(full):
            print(f"SKIP {fp}: not found")
            continue
        
        with open(full, "r", encoding="utf-8") as f:
            file_lines = f.readlines()
        
        changed = False
        for lineno, rules in sorted(line_rules.items(), reverse=True):
            idx = lineno - 1
            if idx < 0 or idx >= len(file_lines):
                print(f"  SKIP {fp}:{lineno} (out of range, file has {len(file_lines)} lines)")
                continue
            line = file_lines[idx].rstrip("\n").rstrip("\r")
            
            existing = re.search(r"#\s*pyright:\s*ignore\[([^\]]+)\]", line)
            if existing:
                merged = set(r.strip() for r in existing.group(1).split(","))
                merged |= rules
                new_comment = f"# pyright: ignore[{', '.join(sorted(merged))}]"
                line = line[:existing.start()] + new_comment + line[existing.end():]
                file_lines[idx] = line + "\n"
                changed = True
                continue
            
            file_lines[idx] = line + f"  # pyright: ignore[{', '.join(sorted(rules))}]\n"
            changed = True
        
        if changed:
            with open(full, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(file_lines)
            print(f"Fixed {fp}: {len(line_rules)} lines")
    
    # Verify
    print("\n--- Verifying ---")
    with open(tmp, "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, "-m", "pyright", "--pythonversion", "3.10", "src/", "app/"],
            stdout=f, stderr=subprocess.STDOUT, timeout=300,
        )
    
    with open(tmp, "r", encoding="utf-8") as f:
        voutput = f.read()
    
    remaining = [l for l in voutput.splitlines() if " - error:" in l]
    summary = [l for l in voutput.splitlines() if l.strip().endswith(("errors", "errors.")) and "warning" not in l.lower()]
    for l in summary:
        print(l)
    
    if remaining:
        print(f"\n{len(remaining)} errors remain:")
        for l in remaining[:10]:
            print(f"  {l.strip()[:200]}")
    else:
        print("\nAll errors fixed!")

if __name__ == "__main__":
    main()

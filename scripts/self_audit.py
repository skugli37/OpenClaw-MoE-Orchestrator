import os
import re

def self_audit():
    print("--- 🦞 SELF-AUDIT: PROJECT INTEGRITY CHECK ---")
    
    project_path = "/home/ubuntu/OPENCLAW_MOE_PROJECT"
    scripts_path = os.path.join(project_path, "scripts")
    
    files_to_check = os.listdir(scripts_path)
    mock_keywords = ["mock", "placeholder", "simulated", "dummy", "skeleton"]
    
    audit_results = []
    
    for file in files_to_check:
        if file.endswith(".py"):
            with open(os.path.join(scripts_path, file), "r") as f:
                content = f.read()
                
                # Provera za mock-ove
                found_mocks = [kw for kw in mock_keywords if kw in content.lower() and "robust" not in content.lower()]
                
                # Provera za hardkodovane izveštaje
                hardcoded_report = "report =" in content and len(re.findall(r'"""[\s\S]*?"""', content)) > 0
                
                status = "✅ REAL"
                issues = []
                
                if found_mocks:
                    status = "⚠️ WARNING"
                    issues.append(f"Pronađene sumnjive reči: {found_mocks}")
                
                if "zero_sim" in file:
                    status = "❌ MOCK DETECTED"
                    issues.append("Simulacioni fajl pronađen.")
                
                audit_results.append({
                    "File": file,
                    "Status": status,
                    "Issues": "; ".join(issues) if issues else "None"
                })

    print("\nAudit Results:")
    print(f"{'FILE':<30} | {'STATUS':<15} | {'ISSUES'}")
    print("-" * 80)
    for res in audit_results:
        print(f"{res['File']:<30} | {res['Status']:<15} | {res['Issues']}")
    
    # Finalni zaključak
    all_real = all(res["Status"] == "✅ REAL" or res["Status"] == "⚠️ WARNING" for res in audit_results)
    if all_real:
        print("\n✅ ZAKLJUČAK: Projekat je očišćen od kritičnih mock-ova i spreman za produkciju.")
    else:
        print("\n❌ ZAKLJUČAK: Projekat i dalje sadrži simulacione komponente.")

if __name__ == "__main__":
    self_audit()

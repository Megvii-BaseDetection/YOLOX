# Security Policy

## Reporting a Vulnerability

### Types of Security Issues
We actively monitor:  
- Code vulnerabilities (RCE, XSS, authentication bypass)  
- Dependency risks (critical vulnerabilities in project dependencies, such as requirements.txt, pyproject.toml, or equivalent files)  
- Configuration flaws (insecure defaults in deployment scripts)  

### Disclosure Channels (Choose one):

1. **Encrypted Email**  
   Contact: `wangfeng19950315@163.com`  
   *Subject format: `[SECURITY] ModuleName - Brief Description`*

2. **GitHub Private Report**  
   Use GitHub's ["Report a vulnerability"](https://github.com/Megvii-BaseDetection/YOLOX/security/advisories) feature  

3. **Reporting Security Issues**  
   Please report security issues using Create new issue: https://github.com/Megvii-BaseDetection/YOLOX/issues/new


## Response Process  
1. **Acknowledgement**  
   - Initial response within **48 business hours**  
2. **Assessment**  
   - Triage using CVSS v3.1 scoring  
3. **Remediation**  
   - Critical (CVSS ≥9.0): Patch within **7 days**  
   - High (CVSS 7-8.9): Patch within **30 days**  
4. **Public Disclosure**  
   - Published via [GitHub Advisories](https://github.com/Megvii-BaseDetection/YOLOX/security/advisories)  
   - CVE assignment coordinated with [MITRE](https://cveform.mitre.org)
  
## Secure Development Practices  
- Always verify hashes when downloading dependencies:  
  ```bash
  sha256sum -c <your-dependency-hash-file>
  ```

# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The QuantScale team takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**finance@vijeth.com**

Include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies based on severity and complexity

### What to Expect

1. **Acknowledgment**: We'll acknowledge receipt of your report within 48 hours
2. **Investigation**: We'll investigate and determine the severity
3. **Updates**: We'll keep you informed of our progress
4. **Fix & Disclosure**: Once fixed, we'll coordinate disclosure timing with you
5. **Credit**: We'll publicly credit you for the discovery (unless you prefer to remain anonymous)

## Security Best Practices

When using QuantScale:

### Data Security
- **Never commit sensitive data** (API keys, credentials, private datasets) to the repository
- Use environment variables for sensitive configuration
- Keep data files in `.gitignore`

### Model Security
- **Validate inputs** before feeding to models
- **Sanitize file paths** when loading checkpoints
- Be cautious with untrusted model checkpoints

### Dependency Security
- Regularly update dependencies: `make update-deps`
- Run security checks: `make security`
- Monitor for CVEs in dependencies

### Docker Security
- Don't run containers as root in production
- Use specific version tags, not `latest`
- Scan images for vulnerabilities before deployment

### Distributed Training
- Use secure communication channels (SSL/TLS) for distributed training
- Authenticate worker nodes in production
- Isolate training environments

## Known Security Considerations

### Model Serialization
- PyTorch `.pt` files can execute arbitrary code when loaded
- Only load model checkpoints from trusted sources
- Consider using ONNX format for inference in production

### Data Loading
- Large files could cause memory issues
- Implement file size limits in production
- Validate data formats before loading

### Ray Tune
- Ray dashboard should not be exposed publicly without authentication
- Use firewall rules to restrict access

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1, 0.1.2) and announced through:
- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Email to reporters

## Scope

This security policy applies to:
- The QuantScale framework code
- Official Docker images
- Scripts and examples in this repository

Out of scope:
- Third-party dependencies (report to respective projects)
- User-specific configurations
- Deployed instances (users are responsible for securing their deployments)

## Attribution

We believe in responsible disclosure and will credit researchers who report valid security issues (with their permission).

## Contact

**Security Contact**: finance@vijeth.com
**Author**: Vithushan Jeyapahan, Vijeth Ltd
**GPG Key**: Available upon request

---

**Thank you for helping keep QuantScale and our users safe!**

Copyright © 2025 Vithushan Jeyapahan, Vijeth Ltd

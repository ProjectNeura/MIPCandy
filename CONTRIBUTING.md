# Contributing to MIPCandy

We welcome contributions to MIPCandy! This guide outlines how you can help improve this medical image processing framework.

## Ways to Contribute

- **Report bugs** and suggest features through [GitHub Issues](https://github.com/ProjectNeura/MIPCandy/issues)
- **Improve documentation** by fixing typos, adding examples, or clarifying concepts
- **Submit code** for bug fixes, new features, or performance improvements
- **Review pull requests** and provide constructive feedback
- **Share your experience** using MIPCandy in medical imaging projects

## Quick Start for Contributors

### 🛠️ Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MIPCandy.git
   cd MIPCandy
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[standard]"
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 📝 Code Guidelines

- **Follow PEP 8** Python style guidelines
- **Add type hints** for function parameters and return values
- **Write docstrings** using Google style format
- **Keep functions focused** - one function, one responsibility
- **Use descriptive variable names** especially for medical imaging contexts

### 🧪 Testing

- Add tests for new functionality in the appropriate test directory
- Ensure existing tests pass before submitting
- Include edge cases and error handling in tests

### 📖 Documentation

- Update relevant documentation for code changes
- Add docstring examples for new functions
- Keep line length under 100 characters
- Use clear, medical-imaging appropriate terminology

## Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Use a descriptive title
   - Explain what your changes do and why
   - Link any relevant issues
   - Add examples if applicable

## Code of Conduct

This project follows the [Python Software Foundation's Code of Conduct](https://www.python.org/psf/conduct/). Please be respectful and constructive in all interactions.

## Getting Help

- **Questions?** Open a [GitHub Discussion](https://github.com/ProjectNeura/MIPCandy/discussions)(Currently NOT available)
- **Bug reports?** Use [GitHub Issues](https://github.com/ProjectNeura/MIPCandy/issues)

## Recognition

Contributors are recognized in our release notes and documentation. Thank you for helping make MIPCandy better for the medical imaging community!

---

*For detailed development guidelines, see our [documentation](docs/README.md).*
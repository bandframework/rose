# BAND SDK v0.2 Community Policy Compatibility for `rose`




**Website:** https://github.com/bandframework/rose \
**Contact:** Contact the rose team via https://github.com/bandframework/rose/issues \
**Icon:** N/a \
**Description:**  The Reduced-Order Scattering Emulator (`rose`) is a user-friendly software for building efficient surrogate models for nuclear scattering. 

### Mandatory Policies

**BAND SDK**
| # | Policy                 |Support| Notes                   |
|---|-----------------------|-------|-------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| `rose` is a Python package and is built using Python's in-built setup tools. [M1 details](#m1-details)|
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly.| Full| A link is placed to the tutorial section of the documentation which contains executable Jupyter notebooks. |
| 3. | Provide a documented, reliable way to contact the development team. |Full| We mention our full support via Github issues. |
| 4. | Come with an open-source license |Full| Uses GPLv3.|
| 5. | Provide a runtime API to return the current version number of the software. |Full| Version can be return via `rose.__version__`. |
| 6. | Provide a BAND team-accessible repository. |Full| Repository at https://github.com/bandframework/rose is fully public. |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained.|Full| All dependencies are managed at runtime and link to externally managed packages. Alternative versions can be explicitly linked to via the PYTHONPATH environment variable. |
| 8. |  Have no hardwired print or IO statements that cannot be turned off. |Full| Aside from warnings and errors in `rose` and external packages, there are no hardwired IO statements in the package itself.|

M1 details <a id="m1-details"></a>: `rose` also has automatic packaging and deployment setup for every tagged release.

### Recommended Policies

| # | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
|**R1.**| Have a public repository. |Full| Repository can be found at https://github.com/bandframework/rose. |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. |Full| Resources are only used while `rose` is loaded in a Python project. The built-in Python garbage collector frees memory when variables are no longer referenced. |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. |Full| Dependencies are handled through the requirements.txt file in the root of the repository. |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Full| Version numbers can be specified in the requirements.txt file. |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Partial| CHANGELOG is handled through Github compare functionality.  Any support is conducted via Github issues, therefore no SUPPORT file is included. |
|**R6.**| Have sufficient documentation to support use and further development.  |Full| Documentation is handled and built using mkdocs. |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. |N/A| No explicit handling of pointers in the Python source for `rose`. |
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. |N/A| None. |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). |Full| `rose` uses the `rose` namespace. |
|**R10.**| Give best effort at portability to key architectures. |Partial| `rose` depends on the portability and availability of the Python interpreter and dependencies. |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. |Full| Prefix is up to the user to decide. No assumption is made in the package build process. |
|**R12.**| All BAND compatibility changes should be sustainable. |Full| None.|
|**R13.**| Respect system resources and settings made by other previously called packages. |Full| None.|
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. |Partial| A partial set of unit tests are currently provided and end-to-end tutorials are provided for the prevailing use cases.|

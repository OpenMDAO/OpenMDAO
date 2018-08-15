"""Key OpenMDAO docutil tools can be imported from here."""

# Doc Utils and Extensions (meant for use by OpenMDAO-dependent repos, e.g. dymos, openaerostruct)
from openmdao.docs._exts import embed_code
from openmdao.docs._exts import embed_options
from openmdao.docs._exts import embed_compare
from openmdao.docs._utils.generate_sourcedocs import generate_docs
from openmdao.docs._utils.preprocess_tags import tag
from openmdao.docs._utils.patch import do_monkeypatch

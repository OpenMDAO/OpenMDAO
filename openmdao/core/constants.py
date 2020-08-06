"""File to define different setup status constants."""

# PRE_SETUP: Newly initialized problem or newly added model.
# POST_CONFIGURE: Configure has been called.
# POST_SETUP: The `setup` method has been called, but vectors not initialized.
# POST_FINAL_SETUP: The `final_setup` has been run, everything ready to run.

PRE_SETUP = 0
POST_CONFIGURE = 1
POST_SETUP = 2
POST_FINAL_SETUP = 3

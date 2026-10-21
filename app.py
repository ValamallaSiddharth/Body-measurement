"""Source-checkout launcher; installed server: bodymeasure-server."""
from bodymeasure.web import app as app, create_app as create_app, main

if __name__ == "__main__":
    main()

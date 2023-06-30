#error received
@app.route("/<string:page_name>")
def page_name(page_name):
    return render_template(page_name)

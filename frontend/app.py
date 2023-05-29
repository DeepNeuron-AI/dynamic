from flask import Flask, render_template
import frontend.dicom_api as dicom_api

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/instances/list")
def list_instances():
    instances = dicom_api.list_store_contents(contents_type=dicom_api.DICOMObjectType.INSTANCES)
    instances = [dicom_api.human_readable_instance(instance) for instance in instances]
    print(instances)
    return render_template("list_instances.html", instances=instances)
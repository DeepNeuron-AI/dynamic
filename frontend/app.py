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

@app.route("/instances/retrieve/<string:study_uid>/<string:series_uid>/<string:instance_uid>")
def retrieve_instance(study_uid: str, series_uid: str, instance_uid: str):
    study_uid = study_uid.replace("-", ".")
    series_uid = series_uid.replace("-", ".")
    instance_uid = instance_uid.replace("-", ".")
    output_file, response = dicom_api.dicomweb_retrieve_instance(study_uid=study_uid, series_uid=series_uid, instance_uid=instance_uid)
    return {
        "outputFile": str(output_file),
        "responseOk": response.ok,
        "responseReason": response.reason,
        "responseStatus": response.status_code,
    }
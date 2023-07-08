from pathlib import Path
from flask import Flask, render_template, request 
import pydicom
import frontend.dicom_api as dicom_api
from frontend.dicom_video import convert_dicom_color, vidwrite
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'
db = SQLAlchemy(app)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    text = db.Column(db.String(500), nullable=False)

with app.app_context():
    db.create_all()


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/instances/list")
def list_instances():
    instances = dicom_api.list_store_contents(contents_type=dicom_api.DICOMObjectType.INSTANCES)
    instances = [dicom_api.human_readable_instance(instance) for instance in instances]
    for instance in instances:
        study_UID = instance["StudyInstanceUID"].replace(".", "-")
        series_UID = instance["SeriesInstanceUID"].replace(".", "-")
        instance_UID = instance["SOPInstanceUID"].replace(".", "-")
        instance["formUrl"] = f"/instances/{study_UID}/{series_UID}/{instance_UID}"

    print(instances)
    return render_template("list_instances.html", instances=instances)

@app.route("/instances/retrieve/<string:study_uid>/<string:series_uid>/<string:instance_uid>")
def retrieve_instance(study_uid: str, series_uid: str, instance_uid: str):
    study_uid = study_uid.replace("-", ".")
    series_uid = series_uid.replace("-", ".")
    instance_uid = instance_uid.replace("-", ".")
    dicom_file, video_file, response = dicom_api.dicomweb_retrieve_instance(study_uid=study_uid, series_uid=series_uid, instance_uid=instance_uid)
    return {
        "outputFile": str(dicom_file),
        "responseOk": response.ok,
        "responseReason": response.reason,
        "responseStatus": response.status_code,
    }

@app.route("/instances/<string:study_uid>/<string:series_uid>/<string:instance_uid>", methods=["GET", "POST"])
def instance_form_view(study_uid: str, series_uid: str, instance_uid: str):
    study_uid = study_uid.replace("-", ".")
    series_uid = series_uid.replace("-", ".")
    instance_uid = instance_uid.replace("-", ".")
    dicom_file, video_file, response = dicom_api.dicomweb_retrieve_instance(study_uid=study_uid, series_uid=series_uid, instance_uid=instance_uid)
    if request.method == "POST":
        text = request.form['text']
        comment = Comment(text=text)
        db.session.add(comment)
        db.session.commit()
        comments = Comment.query.all()
        return render_template("instance_form_view.html", video_filepath=str(video_file), comments=comments)
    return render_template("instance_form_view.html", video_filepath=str(video_file))

if __name__ == "__main__":
    app.run(debug=True)
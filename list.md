


# TO DO ITEMS
- [x] deactivate cloud build once you've done this to save money. View build details and clean it up here: https://cloud.google.com/build/docs/build-push-docker-image or here: https://console.cloud.google.com/appengine/instances?serviceId=default&project=lofty-helix-382115&versionId=20230710t121838
- [x] first confirm if your service is a good fit for cloud run because you thought it was for cloud build but it wasn't. https://cloud.google.com/run/docs/fit-for-run
- [x] start with this https://cloud.google.com/run/docs/quickstarts/deploy-container
- [x] new mdnultrasound email github has been created 
- [x] then do this: https://cloud.google.com/run/docs/quickstarts/deploy-continuously#cloudrun_deploy_continuous_code-python but utilize the tutorial (guide me button)
- [ ] do this https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service using a test main.py python flask file to see if that is the issue of why your code isn't working 
- [ ] try deploying this with your actual frontend.app.py file.
- [ ] connect and deploy cloud run from git instead of from source.
- [ ] connecting cloud run to cloudsql database https://cloud.google.com/run/docs/integrate/connect-to-cloud-sql
- [ ] activate good cloud storage and connect this to your cloud run. look into moving the database and comments into google cloud.
- [ ] connect cloud run to authentication here: https://cloud.google.com/identity-platform
- [ ] ensure DICOM is connected to the server side 
- [ ] look into moving your flask users and accounts into google and combining this with using google accounts for authentication.  
- [ ] turn the .py stuff into cloud functions that can run on google servers instead of my computer 
- [ ] Look into embedding google forms vs adding textboxes. 
-  [ ] add those google forms that allow you to write reports into a specific instance. 
- [ ] add authentication so that people have to login 
- [ ] Await for Norrisa to help fix billing account 
- [ ] add website security? such as implementing https and security on google services. 
- [ ]  deploy the website on google app engine  

# DONE
- [x] remove actual html data in the current html templates by commenting them out and displaying hello worrld
- [x] set up a python flask database to store the web page comments in there 
- [x] you have set up the database and you have been able to put values into the database (I think although i'm not super sure.. like where is this database even kept?? actually it's under instance folder where it says comments.db) and then lastly you just need to display the database onto the web page, which is not working atm. 
- [x] using this for loop given by chat is decent but only displays first few rows, now we just need to have it all be displayed. I need to change the logic so that it displays all the comments immediately which will require a separate python function connected to the database to do this. And then basically whenever we get a POST method, it will reupdate this table to showcase the new comment as well, ie. to add the comment and then simply reload the database. 
- [x] have these instance form views remember the comments that were left on that page. Allow it to have these comments reloaded each time that the instance is opened up. I assume this will require a more complex database backend connection that displays these comments each time that specific instance is opened. 
- [x] activate cloud build api and try running your front end on there.
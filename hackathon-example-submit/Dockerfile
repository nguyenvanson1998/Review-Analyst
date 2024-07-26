# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /shift
COPY requirements.txt /shift

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip install gdown

# STEP 7: Download file weight
# RUN mkdir /Shift/model/
# RUN chmod -R 755 /Shift/model

# RUN gdown https://drive.google.com/drive/folders/1JfiPz5Kf93Tf95L4NxP5vBsfvbUa578P?usp=sharing
# RUN unzip model.zip && rm -rf model.zip
# STEP 8: RUN COMMAND
COPY . .
CMD ["python", "./app.py"]
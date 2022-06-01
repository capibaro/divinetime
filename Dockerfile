FROM python:3.8
ENV PYTHONUNBUFFERED=1
WORKDIR /app/
COPY . .
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple ./wheel/torch-1.7.0+cpu-cp38-cp38-linux_x86_64.whl ./wheel/torchvision-0.8.0-cp38-cp38-linux_x86_64.whl ./wheel/torchaudio-0.7.0-cp38-cp38-linux_x86_64.whl
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple Flask
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple flask-cors
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple gunicorn
ENV FLASK_APP=app.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:4000", "app:app"]
#！/bin/bash
sudo apt-get update -y
sudo apt-get install -y libkcapi-dev
pushd python_demo

if [ -d ".venv" ]; then
  echo 'rm -rf .venv'
  rm -rf .venv
else
  echo "pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple"
  pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  echo "pip3 install virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple"
  pip3 install virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple
  echo "python3 -m virtualenv .venv"
  python3 -m virtualenv .venv
  source .venv/bin/activate
  pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi





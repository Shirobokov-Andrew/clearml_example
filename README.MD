# Работа с ClearML на их же веб-сервере (а не на своём, для которого надо поднимать ClearML Server через Docker Compose)
## Первые шаги
Сначала устанавливаем ClearML:
```
pip install clearml
```
Затем заходим на официальный сайт ClearML https://app.clear.ml/settings/workspace-configuration и регистрируемся там.\
Затем идём в Profile -> Setting -> Workspace и жмём кнопку Create New Credentials.\
Там открывается окно, в котором даны нужные конфиги для ClearML. Копируем их.\
Затем идём в командную строку в вашем проекте, где вы устанавливали ClearML и прописываем:
```
clearml-init
```
ClearML попросит вас вставить скопированный конфиг. Делаем это.\
После этого на вашем компьютере создаётся файл clearml.conf, в котором лежит базовый конфиг ClearML.\
Например, на Windows он лежит в папке 
```
C:\Users\<user>\clearml.conf
```
А на линуксе в
```
~/clearml.conf
```
Этот конфиг затем можно изменять под свои нужды (например, изменить там *api_server*, *web_server*, *files_server* для развёртывания на своём сервере или же, например, добавить credentials для AWS S3) или копировать.\

Привет
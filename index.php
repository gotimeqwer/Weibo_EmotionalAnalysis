<?php
$id = $_POST["text"];
echo $id;
$aa =4;
$bb =6;
$cmd = shell_exec("analyse_text.py $aa $bb");
echo $cmd;
echo $id;
# FSSR
Few Shot Super Resolution


./benchmark -> code avec des metrics pour faire les benchmark type PSNR/SSIM  
./data-processing -> Soupe pour formater les données ou couper les images ou autre

run.py -> le parser CLI qui permet d'éxecuter le code en lui passant les bons arguments. (c'est comme un menu principal)  
utils.py -> les loaders de dataset et petit vérificateur que les fichiers sont bien des images  
finetuner.py -> ébauche de code, pas fini et qui marche pas  
meta.py -> assez important, code de départ = l'implémentation de MAML envoyée sur Slack, j'ai modifié un peu le code pour rajouter des fonctionalités comme les blocks résiduels etc  
FSSR.py -> le code principal. Différentes fonctions qui correspondent à des modes du fichier run.py  
evaluation.py -> code pour faire les benchmarks notamment.  
models.py -> les models vanilla, normalement y'a une implémentation de EDSR mais il faut surement regarder le papier de EDSR pour changer la taille du réseau et les scaling factors pour que ça marche bien.  
loss_functions.py -> pas très utile, c'est des fonctions de perte type perception différentes de MSE  

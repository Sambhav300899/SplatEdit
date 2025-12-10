cd /home/ubuntu/spjain/SplatEdit/data/360_v2/edited_garden/images_2

for f in *.png; do
    mv "$f" "${f%.png}.JPG"
done
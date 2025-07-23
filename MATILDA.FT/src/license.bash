// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


for FILE in *; do
echo ""  | cat - "$FILE" > temp && mv temp "$FILE";
done

for FILE in *; do
echo ""  | cat - "$FILE" > temp && mv temp "$FILE";
done

for FILE in *; do
echo "// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2)."  | cat - "$FILE" > temp && mv temp "$FILE";
done

for FILE in *; do
echo "// Copyright (c) 2023 University of Pennsylvania"  | cat - "$FILE" > temp && mv temp "$FILE";
done

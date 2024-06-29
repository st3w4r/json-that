# Create a plain text schema file
cat << EOF > resources/summary.txt
name: string
command: string
install: string
summary: string
EOF

# Fetch the data from an url and pipe it to jt with a schema file
curl -L jsonthat.com | jt --schema resources/summary.txt

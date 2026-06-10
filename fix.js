const fs = require('fs');
const files = [
    'src/App.jsx',
    'src/components/Sidebar.jsx',
    'src/components/ResultsPanel.jsx',
    'src/components/ViewerPanel.jsx',
    'src/components/ThumbImage.jsx'
];
for (const f of files) {
    let content = fs.readFileSync(f, 'utf8');
    content = content.replace(/\\`/g, '`');
    content = content.replace(/\\\$/g, '$');
    fs.writeFileSync(f, content);
}

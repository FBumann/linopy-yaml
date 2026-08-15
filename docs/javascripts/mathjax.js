// pymdownx.arithmatex in `generic: true` mode rewrites `$...$` / `$$...$$` in
// the source into `\(...\)` / `\[...\]` inside a `.arithmatex` element, so
// those are the delimiters MathJax is told to look for — and it is told to
// look nowhere else, which keeps a `$` in a shell block from becoming math.
//
// A notebook page never passes through arithmatex: mkdocs-jupyter hands mkdocs
// finished HTML, and what `lps.to_markdown` renders into a cell keeps the `$`
// delimiters it was written with. So those are enabled too, and confined the
// same way — `jp-RenderedMarkdown` is the class nbconvert puts on a markdown
// cell and on a Markdown *output*, and on nothing else. A `$` in a notebook's
// code or its printed output sits in `jp-RenderedText`, which is not scanned.
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown",
  },
};

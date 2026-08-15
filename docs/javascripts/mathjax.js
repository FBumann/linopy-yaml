// Two sources of math on this site, and they are spelled differently.
// pymdownx.arithmatex in `generic: true` mode rewrites `$...$` / `$$...$$` in a
// markdown page into `\(...\)` / `\[...\]`; a notebook page never passes through
// it — mkdocs-jupyter hands mkdocs finished HTML, and what `lps.to_markdown`
// rendered into a cell keeps the `$` it was written with. So both delimiter
// sets are enabled.
//
// There is no class restriction, and the notebook is why: MathJax will not
// descend into a subtree that `ignoreHtmlClass` matched to reach a
// `processHtmlClass` element inside it. With `ignoreHtmlClass: ".*|"` — which
// this file used to carry, to keep a stray `$` from becoming math — the
// notebook page typeset none of its nineteen equations, and no arrangement of
// `processHtmlClass` or an explicit `typesetPromise` over the output
// containers changed that. Removing it typesets both kinds of page.
//
// What that guard was protecting is covered anyway: MathJax's default
// `skipHtmlTags` includes `pre` and `code`, so a `$` in a fenced block or an
// inline span is not scanned, and the delimiters only ever meet prose.
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
};

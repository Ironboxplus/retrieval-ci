-- Heading levels for the template's "# Title" / "## Abstract" layout. Typewise
-- traversal runs Header on every block, the abstract in the metadata included,
-- and Meta after it.
--
-- Body sections go one level down (# Introduction -> ## Introduction). Level 6
-- is what the JATS reader gives the captions of tables, figures and
-- supplementary files: they become bold paragraphs, not 7-hash lines.
function Header(h)
  if h.level >= 6 then
    return pandoc.Para({ pandoc.Strong(h.content) })
  end
  h.level = h.level + 1
  h.identifier = ""
  return h
end

-- The abstract's own sections (Background, Methods and findings, ...) are
-- already one down from Header; one more puts them under "## Abstract".
function Meta(m)
  if m.abstract then
    m.abstract = pandoc.Div(m.abstract):walk({
      Header = function(h) h.level = h.level + 1; return h end,
    }).content
  end
  return m
end

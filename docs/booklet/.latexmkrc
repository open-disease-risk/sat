# .latexmkrc configuration file for compiling book.tex using LuaLaTeX and Biber

# Set the default PDF builder to use LuaLaTeX
$pdf_mode = 4;  # 4 = LuaLaTeX

# Commands for LuaLaTeX
$lualatex = 'lualatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Configure Biber for bibliography
$biber = 'biber --bblencoding=utf8 -u -U --output_safechars %O %B';
$bibtex_use = 2;  # Use biber

# Set files to clean
$clean_ext = 'acn acr alg aux bbl bcf blg brf fdb_latexmk glg glo gls idx ilg ind ist lof log lot out run.xml toc dvi nav snm synctex.gz';

# Custom deps
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');

# Configure main file to compile
@default_files = ('book.tex');  # Compile book.tex by default

# Ensure glossaries work if used
sub run_makeglossaries {
    my ($base_name, $path) = fileparse( $_[0] );
    pushd $path;
    my $return = system "makeglossaries $base_name";
    popd;
    return $return;
}

# Set PDF viewer
$pdf_previewer = 'open %O %S';

# Increase the number of compilation runs if needed to resolve references
$max_repeat = 5;

# Add preview feature
$preview_continuous_mode = 1;  # Enable preview-continuous-mode for auto-updates

# For XeLaTeX to be used as option
$xelatex = 'xelatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

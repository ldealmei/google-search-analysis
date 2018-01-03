
# coding: utf-8
# ----------------- Bokeh embedding into HTML -----------------
def modify_html(html, models,script_tag, bokeh_tag) :
    
    # Create bokeh html components
    script, divs = components(models)
    
    # Insert Components into HTML
    # Script
    html_bokeh = re.sub(script_tag,script,html, count = 1)
    # Divs
    for div in divs :
        html_bokeh = re.sub(bokeh_tag,div,html_bokeh, count = 1)
    
    return html_bokeh


def insert_bokeh_into_html(html_file, models, script_tag = "{{bokeh_script}}", bokeh_tag = "{{bokeh}}"):
    """
        models : tuple of models to embed, must be in the order we want them to appear in the HTML
        html_file : html file to modify
        script_tag : tag that marks the place of the bokeh script in the HTML file. There can be only one of these in the HTML. Defaults to {{bokeh_script}}
        bokeh_tag : tag that marks the place of the bokeh plots in the HTML file. There can be muiltiple such tags in the document. Defaults to {{bokeh_script}}
        """
    # Load HTML file
    with open(html_file,'r') as f :
        html = f.read()
    html_bokeh = modify_html(html.decode('utf8'), models,script_tag, bokeh_tag)
    
    html_file_out = re.sub('.html$','_bokeh_embed.html',html_file)
    with open(html_file_out,'w') as f :
        f.write(html_bokeh.encode('utf8'))

# ----------------- From text format to HTML format -----------------
def apply_formating_rules(string, hl_rules, ll_rules) :
    out = string
    
    # High-level processing
    for tag in hl_rules.keys():
        pattern = '^'+tag+'.*'
        if re.search(pattern,string) :
            if tag == '[^{}#\s]': # special case of no tag. Must be interpreted as a simple paragraph
                txt = re.search(pattern,string).group()
                out = re.sub('\[...\]',txt,hl_rules.get(tag))
            
            else :
                s = re.search(pattern,string).group()
                txt = s.split(tag)[1]
                out = re.sub('\[...\]',txt,hl_rules.get(tag))

    # Low-level processing
    for tag in ll_rules.keys():
        if tag == '{{url}}' :
            # special url processing
            pattern = tag + '[^}]*}'
            matches = re.finditer(pattern,out)
            for match in matches :
                m = match.group()
                url_pattern = '}[^{}]*{'
                text_pattern = '{[^{}]*}$'
                url = re.search(url_pattern,m).group()[1:-1]
                txt = re.search(text_pattern,m).group()[1:-1]
                
                repl_txt = re.sub('\[...\]',url,ll_rules.get(tag), count =1)
                repl_txt = re.sub('\[...\]',txt,repl_txt, count =1)
                out = re.sub(pattern,repl_txt,out, count =1)
    
        else :
            # The modification ends at the first space
            pattern = tag+'{[^{}]*}'
            matches = re.finditer(pattern,out)
            for match in matches :
                m = match.group()
                txt = m[len(tag)+1 : -1]
                repl_txt = re.sub('\[...\]',txt,ll_rules.get(tag))
                out = re.sub(pattern, repl_txt,out, count =1)

    return out


def html_format(txt_file, html_template, html_out):
    # List all formating rules. dict keys are the pattern and values are the replacement
    # High-level rules are those that format at the paragraph level
    hl_rules = {'{{title}}' :'<h2 class="w3-wide">[...]</h2>',
                '{{subtitle}}' :'<h4 class="w3-opacity">[...]</h4>',
                '{{section}}' :'<h3 class="w3-justify">[...]</h3>',
                '{{caption}}':'<p class="w3-opacity w3-center"><i>[...]</i></p>',
                '{{p}}' :'<p class="w3-justify">[...]</p>',
                '[^{}#\s]' :'<p class="w3-justify">[...]</p>',
                '{{img}}':'<img src="[...]" class="w3-round" alt="[missing image]" width="800px">',
                '#' :''}
    # Low-level rules are the HTML modification that occur within a paragraph
    ll_rules = {'{{i}}' : '<i>[...]</i>',
                '{{b}}' : '<b>[...]</b>',
                '{{url}}' : '<a href="[...]">[...]</a>'}

    content_tag = '{{text_content}}'

    # Read text file and html template
    with open(txt_file, 'r') as f :
        text = f.read()
    with open(html_template, 'r') as f :
        html_template = f.read()

    text_split = text.split('\n')
    
    #     ##[...]eof : ignore everything that follows
    try :
        idx_ignore_rest = text_split.index('##')
        text_split= [line for line in text_split if (line!='' and text_split.index(line)< idx_ignore_rest)]
    except :
        text_split= [line for line in text_split if line]
    
    # Apply all rules
    text_split_formated = []
    for string in text_split :
        text_split_formated.append(apply_formating_rules(string, hl_rules, ll_rules))

    text_formated ='\n'.join(text_split_formated)
    
    # Insert into html
    html = re.sub(content_tag,text_formated,html_template)
    
    # Save html file
    with open(html_out, 'w') as f :
        f.write(html)


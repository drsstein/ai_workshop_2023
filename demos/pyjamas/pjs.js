// wrapper around pyodide for easy interaction with python from javascript
var console = {log: console.log,
               warn: console.warn,
               error: console.error};

var pjs = (function() {
    // private stuff
    
    // wrapper for dom element that displays python statements and output
    class TextOutput {
        cmd_prefix = ">>>"; //prefix for command prints
        
        constructor (dom_id=null) {
            this.dom_id = dom_id;
            this.flush = false;
        }
        // prints a message to the output DOM element
        // args:
        //   message (str): message to be printed; accepts multiple line 
        //                  separated by "\n"
        //   flush (bool):  clear output before printing the *next* message.
        //   prefix (str):  every printed row will start with this string.
        print(message, {flush = false, prefix = "", is_error = false} = {}) {
            if (this.dom_id == null) {
                // default to console output
                console.log(message);
            } else {
                var output = document.getElementById(this.dom_id);
                var html = output.innerHTML;
                if (this.flush) {
                    html = ""
                }
                var new_html = "";
                var lines = (message + "").split("\n");
                lines.forEach(function (line) {
                    new_html += prefix + line + '<br>';
                });
                if (is_error) {
                    new_html = "<span style='color:red;'>" + new_html + "</span>";
                }
                output.innerHTML = html + new_html;
                output.scrollTo(0, output.scrollHeight);
                this.flush = flush;
            }
        }
        
        // return callback print function with proper object binding
        get_callback(options) {
            return function(message){
                return this.print(message, options);
            }.bind(this);
        }
        
        make_console(options) {
            console.log = this.get_callback(options);
        }
    };
    
    class PythonJS {
        constructor() {
            this.stdout = new TextOutput();
            //this.print = this.stdout.print;
        }
        
        // run a piece of python code; print result (default:true)
        // args:
        // script (str): python script to execute (can be multi-line string).
        // auto_install (bool): install all modules found in import statements.
        run = async function(script, {print_script = false, 
                                      print_result = false,
                                      auto_install = true} = {}) {
            if (print_script) {
                console.log(script);
            }
            try {
                // parse script for "!pip install" and import statements
                var lines = (script + "").split("\n");
                var pip_installs = [];
                var imports = [];
                var py_script = "";
                const regex_pip = /\t*!pip install/;
                const regex_from = /\t*from/
                const regex_import = /\t*import/;
                await lines.forEach(async function(line) {
                    if (regex_pip.test(line)) {
                        //catch and install packages
                        var line = line.split("!pip install")[1];
                        var words = line.split(/, \t/);
                        pip_installs = pip_installs.concat(words);
                    } else {
                        // before doing anything else, append the line to the
                        // executed script - else it might be inserted in wrong
                        // order due to async function calls.
                        py_script += line + "\n";
                        var module = null;
                        if (regex_from.test(line)) {
                            var rest = line.split("from ")[1];
                            module = rest.split(/[\s. \t]+/)[0];
                        } else if (regex_import.test(line)) {
                            var rest = line.split("import ")[1];
                            module = rest.split(/[\s. \t]+/)[0];
                        }
                        // try executing line and append if it fails
                        if (module != null) {
                            // try executing import and only install if import fails
                            // this avoids trying to install standard libraries that
                            // would throw a micropip error later.
                            try {
                                await this.pyodide.runPython(line);
                            } catch(err) {
                                imports.push(module);
                            }
                        }
                    }
                }.bind(this));
                // install modules
                if (auto_install) {
                    pip_installs = pip_installs.concat(imports);
                }
                if (pip_installs.length > 0) {
                    await this.micropip.install(pip_installs);
                    console.log("--- installation complete ---");
                }
                // run python script
                const result = await this.pyodide.runPython(py_script);
                if (print_result) {
                    console.log(result);
                }
                return result;
            } catch(err) {
                console.error(err.message);
            }
        }
        
        fetch_and_run = async function(url, args = {}) {
			var response = await fetch(url, {cache: "reload", headers: {"Content-Type": "application/text"}});
			var text = await response.text();
			await this.run(text, args);
		}
		
		// initialise python interface
        // stdout: DOM ID of stdout
        init = async function(stdout=null, stdout_args = {}) {
            this.stdout.dom_id = stdout;
			stdout_args['prefix'] = "[pjs] ";
            this.stdout.make_console(stdout_args);
            console.log("initializing python...");
            this.pyodide = await loadPyodide();
            await this.pyodide.loadPackage("micropip", {messageCallback: this.stdout.get_callback({prefix: "[pyodide]"})});
            this.micropip = this.pyodide.pyimport("micropip");
            const default_script = `
                import sys
                sys.version
            `;
            await this.run(default_script, {auto_install:false, print_result: true});
        }
        
        get = function(var_name) {
            // split variable by '.' and do recursive access
            const objects = var_name.split(".");
            var output = this.pyodide.globals.get(objects[0]);
            for (let i=1; i < objects.length; i++) {
                output = output[objects[i]];
            }
            return output;
        }
        
        code_cell(dom_id, cell_id) {
            // create a code cell inside a DOM element
            
            // input text area
            var input_dom = document.createElement("textarea");
            input_dom.id = cell_id + "_input";
            input_dom.rows = 8;
            input_dom.cols = 80;
            input_dom.style.fontFamily = "monospace";
            
            // output area
            var output_dom = document.createElement("p");
            output_dom.id = cell_id + "_output";
            output_dom.style.fontFamily = "monospace";
            
            // load script button
            var file_dom = document.createElement("input");
            file_dom.type = "file";
            file_dom.onchange = function() {
                if (this.files && this.files[0]) {
                    var file = this.files[0];
                    var reader = new FileReader();
                    reader.addEventListener('load', function (event) {
                      input_dom.textContent = event.target.result;
                    });
                    reader.readAsBinaryString(file);
                }   
            };
            
            // output object
            var output = new TextOutput(output_dom.id);
            
            async function runScript() {
                output.make_console({flush:true});
                await this.run(input_dom.value, {print_script: false, print_result:true});
                this.stdout.make_console();
            }
            
            input_dom.onkeydown = function (event) {
                if (event.key === "Enter" && event.shiftKey) {
                    event.preventDefault();
                    runScript.bind(this)();
                };
            }.bind(this);
            
            // run button
            var run_dom = document.createElement("button");
            run_dom.onclick = runScript.bind(this);
            run_dom.appendChild(document.createTextNode("Run"));
            
            // delete button
            var child = document.createElement("div");
            var parent = document.getElementById(dom_id);
            parent.appendChild(child);
            
            var del_dom = document.createElement("button");
            del_dom.onclick = () => parent.removeChild(child);
            del_dom.appendChild(document.createTextNode("X"));
            
            // up button
            var up_dom = document.createElement("button");
            up_dom.onclick = function() {
                var my_position = Array.from(parent.children).indexOf(child);
                parent.insertBefore(child, parent.children[Math.max(0, my_position-1)]);
            };
            up_dom.appendChild(document.createTextNode("^"));
            
            // down button
            var down_dom = document.createElement("button");
            down_dom.onclick = function() {
                var my_position = Array.from(parent.children).indexOf(child);
                parent.insertBefore(child, parent.children[my_position+2]);
            };
            down_dom.appendChild(document.createTextNode("v"));

            child.appendChild(run_dom);
            child.appendChild(up_dom);
            child.appendChild(down_dom);
            child.appendChild(del_dom);
            child.appendChild(file_dom);
            child.appendChild(document.createElement("br"));
            child.appendChild(input_dom);
            child.appendChild(document.createElement("br"));
            child.appendChild(output_dom);
        }
        
        terminal = function(dom_id) {
            // manage a list of code cells inside a dom element
            
            const parent = document.getElementById(dom_id)
            parent.innerHTML = "<p>Click Run or press (Shift + Enter) to execute a cell.</p>";
            var cell_div = document.createElement("div");
            cell_div.id = dom_id + "_cells";
            parent.appendChild(cell_div);
            
            var num_cells = 0;
            var append = document.createElement("button");
            append.onclick = function() {
                this.code_cell(cell_div.id, num_cells++);
            }.bind(this);
            append.appendChild(document.createTextNode("New Cell"));
            parent.appendChild(append);
            
            // create first cell
            this.code_cell(cell_div.id, num_cells++);
        }

    };
    
    // public
    return new PythonJS();
  })();
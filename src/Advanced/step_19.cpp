/**
 * @file Handle input parameters. Convert from deal.II intermediate format
 * to other graphical formats.
 */

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

#include <list>
#include <iostream>
#include <fstream>
#include <stdlib.h>

namespace step19
{
    using namespace dealii;

    ParameterHandler prm;
    std::vector<std::string> inputFileNames;
    std::string outputFile;
    std::string outputFormat;

    void printUsageMessage ()
    {
        static const char *message =
                "\n"
                "Converter from deal.II intermediate format to other graphics formats.\n"
                "\n"
                "Usage:\n"
                "    ./step-19 [-p parameter_file] list_of_input_files \n"
                "              [-x output_format] [-o output_file]\n"
                "\n"
                "Parameter sequences in brackets can be omitted if a parameter file is\n"
                "specified on the command line and if it provides values for these\n"
                "missing parameters.\n"
                "\n"
                "The parameter file has the following format and allows the following\n"
                "values (you can cut and paste this and use it for your own parameter\n"
                "file):\n"
                "\n";
        std::cout << message;

        prm.print_parameters(std::cout, ParameterHandler::Text);
    }

    void declareParameters ()
    {
        prm.declare_entry("Output file", "", Patterns::Anything(),
                          "The name of the output file to be generate");

        DataOutInterface<1>::declare_parameters(prm);

        prm.declare_entry("Dummy iterations", "42",
                         Patterns::Integer(1, 1000),
                         "A dummy parameter asking for an integer");

        prm.enter_subsection("Dummy subsection");
        {
            prm.declare_entry ("Dummy generate output", "true",
                               Patterns::Bool(),
                               "A dummy parameter that can be fed with either "
                               "\'true\' or \'false\'");
            prm.declare_entry ("Dummy color of output", "red",
                               Patterns::Selection("red|black|blue"),
                               "A dummy parameter that shows how one can define a "
                               "parameter that can be assigned values from a finite "
                               "set of values");
        }
        prm.leave_subsection();
    }

    void parseCommandLine(const int argc, char * const *argv)
    {
        if (argc < 2) {
            printUsageMessage();
            ::exit(1);
        }

        std::list<std::string> args(argc);
        std::copy(argv, argv + argc, args.begin());

        while (!args.empty()) {
            if (args.front() == std::string("-p")) {
                if (args.size() == 1) {
                    std::cerr << "Error: flag '-p' must be followed by the "
                              << "name of a parameter file."
                              << std::endl;
                    printUsageMessage ();
                    ::exit (1);
                }
                args.pop_front();
                const std::string parameterFile = args.front();
                args.pop_front();

                prm.read_input(parameterFile);

                if (outputFile == "") {
                    outputFile = prm.get("Output file");
                }

                if (outputFormat == "") {
                    outputFormat = prm.get("Output format");
                }

                prm.enter_subsection("Dummy subsection");
                {
                    prm.get_bool("Dummy generate output");
                }
                prm.leave_subsection();
            }
            else if (args.front() == std::string("-x")) {
                if (args.size() == 1)
                {
                    std::cerr << "Error: flag '-o' must be followed by the "
                              << "name of an output file."
                              << std::endl;
                    printUsageMessage ();
                    ::exit(1);
                }
                args.pop_front ();
                outputFile = args.front();
                args.pop_front();
            }
            else {
                inputFileNames.push_back(args.front());
                args.pop_front();
            }
        }
        if (inputFileNames.empty()) {
            std::cerr << "Error: No input file specified." << std::endl;
            printUsageMessage();
            ::exit(1);
        }
    }

    template <int dim, int spacedim>
    void doConvert()
    {
        DataOutReader<dim, spacedim> mergedData;
        {
            std::ifstream input(inputFileNames[0]);
            AssertThrow (input, ExcIO());

            mergedData.read(input);
        }

        for (unsigned int i=1; i<inputFileNames.size(); ++i)
        {
            std::ifstream input (inputFileNames[i].c_str());
            AssertThrow (input, ExcIO());
            DataOutReader<dim,spacedim> additionalData;
            additionalData.read (input);
            mergedData.merge (additionalData);
        }

        std::ofstream outputStream (outputFile);
        AssertThrow (outputStream, ExcIO());

        const DataOutBase::OutputFormat format =
                DataOutBase::parse_output_format(outputFormat);
        mergedData.write(outputStream, format);
    }

    void convert ()
    {
        AssertThrow(!inputFileNames.empty(), ExcMessage("No input files specified."));

        std::ifstream input(inputFileNames[0]);
        AssertThrow(input, ExcIO());

        const std::pair<size_t, size_t> dimensions =
                DataOutBase::determine_intermediate_format_dimensions(input);

        switch (dimensions.first)
        {
            case 1:
                switch (dimensions.second)
                {
                    case 1:
                        doConvert <1, 1>();
                        return;
                    case 2:
                        doConvert<1, 2>();
                        return;
                }
                AssertThrow(false, ExcNotImplemented());
            case 2:
                switch (dimensions.second)
                {
                    case 2:
                        doConvert<2, 2>();
                        return;
                    case 3:
                        doConvert<2, 3>();
                        return;
                    AssertThrow(false, ExcNotImplemented());
                }
            AssertThrow(false, ExcNotImplemented());
        }
    }
}

int main (int argc, char ** argv)
{
    try {
        using namespace step19;

        declareParameters();
        parseCommandLine(argc, argv);

        convert();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    };
    return 0;
}
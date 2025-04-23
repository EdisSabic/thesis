
robostudio_questions = [
    "What is the difference between a station and a project?",
    "What is RobotStudio and what are its primary functions?",
    "What are the system requirements for installing RobotStudio?",
    "How can you activate a RobotStudio license, and what are the different types of licenses available?",
    "What steps are involved in connecting a PC to a robot controller using RobotStudio?",
    "How do you manage user rights and write access on an IRC5 controller?",
    "What are the key features of the RAPID editor in RobotStudio?",
    "How can you create and configure a virtual controller in RobotStudio?",
    "What is the purpose of the Smart Components in RobotStudio, and how can they be used?",
    "How do you set up and run a simulation in RobotStudio?",
    "What are the steps to create a collision-free path between two targets in RobotStudio?",
    "How can you configure and use the I/O Simulator in RobotStudio?",
    "What are the different types of joints available in RobotStudio for physics simulations?",
    "How can you use the OPC UA Client Smart Component for virtual commissioning in RobotStudio?",
    "What are the steps to create and use a custom instruction template in RobotStudio?",
    "How can you save and load RAPID programs and modules in RobotStudio?"
]

rw_rapid_questions = [
    "What is a suitable instruction for linear movement?",
    "What is the purpose of the AccSet instruction in RAPID programming?",
    "How does the ActEventBuffer instruction affect the execution of robot movements?",
    "Explain the usage of the AliasCamera instruction.",
    "What are the arguments required for the Add instruction and what does it do?",
    "Describe the function of the BitClear instruction.",
    "How can the BookErrNo instruction be used to handle custom errors in RAPID?",
    "What is the difference between ConfJ and ConfL instructions?",
    "How does the ContactL instruction work and what is its primary use?",
    "Explain the purpose of the CorrCon and CorrDiscon instructions.",
    "What does the DeactUnit instruction do and when should it be used?",
    "Describe the process and arguments for the EOffsOn instruction.",
    "What is the significance of the MoveL instruction in RAPID programming?",
    "How does the PDispOn instruction affect robot movements?",
    "Explain the usage and importance of the WaitLoad instruction.",
    "What are the limitations of the CapL instruction and how can errors be handled?"
]

overview_rapid_questions = [
    "What is the purpose of the RAPID programming language as described in the document?",
    "How are instructions and functions represented in RAPID syntax?",
    "What are the three types of routines in RAPID, and how do they differ?",
    "Explain the concept of 'modules' in RAPID programming. What are the differences between program modules and system modules?",
    "Describe the role and structure of data declarations in RAPID. What are the different kinds of data that can be declared?",
    "What is the significance of the tool center point (TCP) in robot programming, and how is it defined?",
    "How does RAPID handle motion instructions, and what are the different types of interpolation methods available?",
    "What are World Zones, and how are they used in RAPID programming to enhance robot safety and functionality?",
    "Explain the concept of 'soft servo' in RAPID. How does it affect the robot's movement?",
    "What is the purpose of the 'UNDO' handler in RAPID routines, and when is it executed?",
    "How does RAPID manage error recovery, and what instructions are used to handle errors within a program?",
    "Describe the process and importance of calibration in RAPID programming. What types of calibration methods are mentioned?",
    "What are the key features of multitasking in RAPID, and how does it benefit robot programming?",
    "How does RAPID support communication with external devices and systems? What are some of the communication instructions provided?",
    "What is the role of the 'configuration control' in RAPID, and how does it ensure the robot follows the correct path and orientation?"
]

rw_system_parameters_questions = [
    "What is the purpose of the RobotWare software's Connected Services functionality?",
    "How can the parameter Enabled affect the connection to ABB Connected Services Cloud?",
    "What types of network connections can be defined using the Connection Type parameter?",
    "What is the role of the Internet Gateway IP parameter in the Connected Services configuration?",
    "How does the Proxy Used parameter influence the connection settings?",
    "What does the Server Polling parameter control in the Connected Services setup?",
    "What is the significance of the Debug Mode parameter in troubleshooting connectivity issues?",
    "How does the Enable 3G connection parameter affect the Connected Services Gateway 3G module?",
    "What is the function of the Roaming parameter in the Connected Services Gateway 3G configuration?",
    "How is the Access Point Name parameter used in the Connected Services Gateway 3G setup?",
    "What does the Enable Wi-Fi connection parameter do in the Connected Services Gateway Wi-Fi module?",
    "How does the SSID parameter affect the Wi-Fi connection settings?",
    "What is the purpose of the Security parameter in the Connected Services Gateway Wi-Fi configuration?",
    "How does the State parameter influence the Connected Services Gateway Wired module?",
    "What role does the IP Address parameter play in the Connected Services Gateway Wired setup?"
]

all_questions = []
all_questions.extend(robostudio_questions)
all_questions.extend(rw_rapid_questions)
all_questions.extend(overview_rapid_questions)
all_questions.extend(overview_rapid_questions)

import os

density = ["0.10", "0.15", "0.20", "0.40", "0.80"]

for nov in range(30,39):
	for den in density:
		mat = "int/"+str(nov)+"_"+den+"_0"
		
		cpu_cmd = "./a.out -m " + mat + " -c -t16 > ./Exps/cpu16_" + str(nov) + "_" + den
		os.system(cpu_cmd)

		if den in ["0.10", "0.15", "0.20"]:
			cpu_cmd = "./a.out -m " + mat + " -s -c -t16 > ./Exps/cpu16_sparse_" + str(nov) + "_" + den
			os.system(cpu_cmd)

		for gpu_algo in range(1,5):
			gpu_cmd = "./a.out -m " + mat + " -g -p" + str(gpu_algo) + " > ./Exps/gpu" + str(gpu_algo) + "_" + str(nov) + "_" + den
			os.system(gpu_cmd)
			if den in ["0.10", "0.15", "0.20"]:
				gpu_cmd = "./a.out -m " + mat + " -s -g -p" + str(gpu_algo) + " > ./Exps/gpu" + str(gpu_algo) + "_sparse_" + str(nov) + "_" + den
				os.system(gpu_cmd)


cmd = "./a.out -m int/40_0.20_0 -g -p4 -b > ./ExpsAppr/exact40_0.20"
cmd = "./a.out -m int/40_0.60_0 -g -p4 -b > ./ExpsAppr/exact40_0.60"
cmd = "./a.out -m int/40_0.85_0 -g -p4 -b > ./ExpsAppr/exact40_0.85"

cmd = "./a.out -m int/40_0.20_0 -c -p1 -b -a -x400000 > ./ExpsAppr/cpu400k_rass40_0.20"
cmd = "./a.out -m int/40_0.60_0 -c -p1 -b -a -x400000 > ./ExpsAppr/cpu400k_rass40_0.60"
cmd = "./a.out -m int/40_0.85_0 -c -p1 -b -a -x400000 > ./ExpsAppr/cpu400k_rass40_0.85"

cmd = "./a.out -m int/40_0.20_0 -c -p2 -b -a -x400000 -y4 -z5 > ./ExpsAppr/cpu400k_appr40_0.20"
cmd = "./a.out -m int/40_0.60_0 -c -p2 -b -a -x400000 -y4 -z5 > ./ExpsAppr/cpu400k_appr40_0.60"
cmd = "./a.out -m int/40_0.85_0 -c -p2 -b -a -x400000 -y4 -z5 > ./ExpsAppr/cpu400k_appr40_0.85"

cmd = "./a.out -m int/40_0.20_0 -g -p1 -b -a -x4000000 > ./ExpsAppr/gpu4m_rass40_0.20"
cmd = "./a.out -m int/40_0.60_0 -g -p1 -b -a -x4000000 > ./ExpsAppr/gpu4m_rass40_0.60"
cmd = "./a.out -m int/40_0.85_0 -g -p1 -b -a -x4000000 > ./ExpsAppr/gpu4m_rass40_0.85"

cmd = "./a.out -m int/40_0.20_0 -g -p2 -b -a -x4000000 -y4 -z5 > ./ExpsAppr/gpu4m_appr40_0.20"
cmd = "./a.out -m int/40_0.60_0 -g -p2 -b -a -x4000000 -y4 -z5 > ./ExpsAppr/gpu4m_appr40_0.60"
cmd = "./a.out -m int/40_0.85_0 -g -p2 -b -a -x4000000 -y4 -z5 > ./ExpsAppr/gpu4m_appr40_0.85"

cmd = "./a.out -m int/40_0.20_0 -c -p1 -b -a -s -x400000 > ./ExpsAppr/cpu400k_sp_rass40_0.20"
cmd = "./a.out -m int/40_0.20_0 -c -p2 -b -a -s -x400000 -y4 -z5 > ./ExpsAppr/cpu400k_sp_appr40_0.20"
cmd = "./a.out -m int/40_0.20_0 -g -p1 -b -a -s -x4000000 > ./ExpsAppr/gpu4m_sp_rass40_0.20"
cmd = "./a.out -m int/40_0.20_0 -g -p2 -b -a -s -x4000000 -y4 -z5 > ./ExpsAppr/gpu4m_sp_appr40_0.20"
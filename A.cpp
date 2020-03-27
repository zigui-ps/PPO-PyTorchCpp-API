#include "Environment/GymEnvironment.h"
#include "RL/Policy.h"

void PythonInit(int argc, char *argv[]){
    Py_Initialize();
	wchar_t **argw = new wchar_t*[argc];
	for(int i = 0; i < argc; i++) argw[i] = Py_DecodeLocale(argv[i], NULL);
    PySys_SetArgv(argc, argw);
}

AgentInterfacePtr PPO_agent_with_param(GymEnvironmentPtr env, std::vector<int> actor_size, double actor_lr, \
		std::vector<int> critic_size, double critic_lr, double critic_decay, double gamma, double lamda, int steps, int batch_size){
	int o_size = env->observationSize;
	int a_size = env->actionSize;
	actor_size.insert(actor_size.begin(), o_size); actor_size.insert(actor_size.end(), a_size);
	DeepNetworkPtr actor_network = DeepNetworkPtr(new DeepNetwork(actor_size));
	AdamPtr actor_opt = AdamPtr(new torch::optim::Adam(actor_network->parameters(), actor_lr));
	DistributionInterfacePtr dist = DistributionInterfacePtr(new GaussianDistribution(torch::ones(a_size)));
	ActorPtr actor = ActorPtr(new Actor(actor_network, actor_opt, dist));

	critic_size.insert(critic_size.begin(), o_size); critic_size.insert(critic_size.end(), 1);
	DeepNetworkPtr critic_network = DeepNetworkPtr(new DeepNetwork(critic_size));
	torch::optim::AdamOptions critic_options(critic_lr); critic_options.weight_decay(critic_decay);
	AdamPtr critic_opt = AdamPtr(new torch::optim::Adam(critic_network->parameters(), critic_options));
	CriticPtr critic = CriticPtr(new Critic(critic_network, critic_opt));

	StateModifierPtr state_modifier = StateModifierPtr(new ClassicModifier(o_size));

	return AgentInterfacePtr(new PPOAgent(env, actor, critic, state_modifier, gamma, lamda, steps, batch_size));
}

void train(AgentInterfacePtr agent, int train_step, torch::Device device){
	mkdir("save_model", 0775);
	for(int i = 0; i < train_step; i++){
		agent->train(5, device, true);
		printf("train fin\n");
		std::stringstream name;
		name << "./save_model/" << i;
		tinyxml2::XMLDocument doc;
		tinyxml2::XMLElement* root = doc.NewElement("RLModel");
		doc.LinkEndChild(root);
		root->LinkEndChild(agent->get_xml(name.str(), doc));
		doc.SaveFile((name.str() + ".xml").c_str());
		std::cout << "saved at " << name.str() << ".xml\n";
	}
}

void demo(GymEnvironmentPtr env, AgentInterfacePtr agent, torch::Device device){
	agent->to(device);
	torch::Tensor state = env->reset();
	while(1){
		env->render();
		torch::Tensor next_state, action; double reward; int done, tl;
		action = agent->next_action_nodist(state);
		env->step(action, next_state, reward, done, tl);
		state = next_state;
		if(done) printf("END\n");
	}
}

static std::string load_model;
static torch::Device device = torch::kCPU;
static int train_step = 0;
static std::string env_type;

void ParseArgs(int argc, char *argv[]){
	for(int i = 0; i < argc; i++){
		if(strcmp(argv[i], "--load_model") == 0) load_model = argv[++i];
		if(strcmp(argv[i], "--train_step") == 0) train_step = atoi(argv[++i]);
		if(strcmp(argv[i], "--gpu") == 0){
			if(torch::cuda::is_available()) device = torch::kCUDA;
			else printf("CUDA is not available\n");
		}
		if(strcmp(argv[i], "--env") == 0) env_type = argv[++i];
	}
}

int main(int argc, char *argv[])
{
	PythonInit(argc, argv);
	ParseArgs(argc, argv);

	GymEnvironmentPtr env = GymEnvironmentPtr(new GymEnvironment(env_type.c_str(), device));
	AgentInterfacePtr agent = PPO_agent_with_param(env, {128, 128}, 1e-4, {128, 128}, 1e-4, 7e-4, 0.994, 0.99, 4096, 80);
	//AgentInterfacePtr agent = Vanila_agent_with_param(env, {128, 128}, 1e-4, {128, 128}, 1e-4, 7e-4, 0.994, 2048, 32);
	if(load_model != ""){
		tinyxml2::XMLDocument doc;
		if(doc.LoadFile(load_model.c_str())) return !printf("%s not exist\n", load_model.c_str());
		agent->set_xml(doc.RootElement());
	}
	train(agent, train_step, device);
	demo(env, agent, device);

	if (Py_FinalizeEx() < 0) {
		return 120;
	}
	return 0;
}// */

/*
AgentInterfacePtr Vanila_agent_with_param(GymEnvironmentPtr env, std::vector<int> actor_size, double actor_lr, \
		std::vector<int> critic_size, double critic_lr, double critic_decay, double gamma, int steps, int batch_size){
	int o_size = env->observationSize;
	int a_size = env->actionSize;
	actor_size.insert(actor_size.begin(), o_size); actor_size.insert(actor_size.end(), a_size);
	DeepNetworkPtr actor_network = DeepNetworkPtr(new DeepNetwork(actor_size));
	AdamPtr actor_opt = AdamPtr(new torch::optim::Adam(actor_network->parameters(), actor_lr));
	DistributionInterfacePtr dist = DistributionInterfacePtr(new GaussianDistribution(torch::ones(a_size)));
	ActorPtr actor = ActorPtr(new Actor(actor_network, actor_opt, dist));

	critic_size.insert(critic_size.begin(), o_size); critic_size.insert(critic_size.end(), 1);
	DeepNetworkPtr critic_network = DeepNetworkPtr(new DeepNetwork(critic_size));
	torch::optim::AdamOptions critic_options(critic_lr); critic_options.weight_decay(critic_decay);
	AdamPtr critic_opt = AdamPtr(new torch::optim::Adam(critic_network->parameters(), critic_options));
	CriticPtr critic = CriticPtr(new Critic(critic_network, critic_opt));

	StateModifierPtr state_modifier = StateModifierPtr(new ClassicModifier());

	return AgentInterfacePtr(new VanilaAgent(env, actor, critic, state_modifier, gamma, steps, batch_size));
} // */ 

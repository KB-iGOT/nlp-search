@Library('deploy-conf') _
node() {
    try {
        String ANSI_GREEN = "\u001B[32m"
        String ANSI_NORMAL = "\u001B[0m"
        String ANSI_BOLD = "\u001B[1m"
        String ANSI_RED = "\u001B[31m"
        String ANSI_YELLOW = "\u001B[33m"

        ansiColor('xterm') {
                stage('Checkout') {
                    if (!env.hub_org) {
                        println(ANSI_BOLD + ANSI_RED + "Uh Oh! Please set a Jenkins environment variable named hub_org with value as registery/sunbidrded" + ANSI_NORMAL)
                        error 'Please resolve the errors and rerun..'
                    } else
                        println(ANSI_BOLD + ANSI_GREEN + "Found environment variable named hub_org with value as: " + hub_org + ANSI_NORMAL)
                }
                cleanWs()
                checkout scm
                commit_hash = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
	        build_tag = sh(script: "echo " + params.github_release_tag.split('/')[-1] + "_" + commit_hash + "_" + env.BUILD_NUMBER, returnStdout: true).trim()
                echo "build_tag: " + build_tag

        stage('Install Dev Dependencies') {
                sh '''
                    pip install --break-system-packages \
                        pytest>=8.3.0 \
                        pytest-cov>=5.0.0 \
                        pytest-asyncio>=0.25.0 \
                        httpx>=0.27.2 \
                        anyio[trio]>=4.8.0
                '''
            }

                stage('Test & Coverage') {
                    // Fails pipeline if coverage < 80% (fail_under=80 in pyproject.toml)
                    sh '''
                        pytest \
                            --cov=src \
                            --cov-report=xml:coverage.xml \
                            --cov-report=term-missing \
                            --cov-config=pyproject.toml \
                            -v
                    '''
                }

                stage('SonarQube Analysis') {
                    // Runs only after Test & Coverage (coverage.xml must exist)
                    // Credentials injected via Jenkins env — never hardcoded
                    withSonarQubeEnv('SonarQube') {
                        sh '''
                            sonar-scanner \
                                -Dsonar.host.url=${SONAR_HOST_URL} \
                                -Dsonar.token=${SONAR_TOKEN}
                        '''
                    }
                }

                stage('Quality Gate') {
                    // Blocks deployment if SonarQube Quality Gate is red
                    timeout(time: 5, unit: 'MINUTES') {
                        waitForQualityGate abortPipeline: true
                    }
                }

        stage('Build') {
                env.NODE_ENV = "build"
                print "Environment will be : ${env.NODE_ENV}"
                sh('chmod 777 build.sh')
                sh("bash -x build.sh ${build_tag} ${env.NODE_NAME} ${docker_server}")
            }
                        stage('ArchiveArtifacts') {
                    archiveArtifacts "metadata.json"
                    currentBuild.description = "${build_tag}"
                }

      }
        
	}
    catch (err) {
        currentBuild.result = "FAILURE"
        throw err
    }
    finally {
      //  email_notify()
    }
}

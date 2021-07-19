# 윈도우 환경에서 딥러닝 실습

노트북 파일안에 다음과 같은 리눅스 명령어는 윈도우에서 실행되지 않는다. 
```
!wget http://some/url
!ls -al
```

윈도우즈에서 이러한 리눅스 명령어 대신 실습하기 위한 가이드.


<br>

# 환경 설정

다음과 같은 사항이 설치 혹은 설정되어 있어야 한다.
- GPU
- 아나콘다
- TensorFlow

윈도우에서 환경 설정은 다음을 참조한다. [python/install_anaconda.pdf](python/install_anaconda.pdf)


<br>


# 리눅스 명령어 대신할 방법

대상 리눅스 명령어

- ls -al
- wget
- tar or zip
- git
- nvidia-smi


<br>

파일 다운로드, 압축 풀기등은 실습 노트북이 있는 폴더에서 실행하는 것을 전제로 한다.


<br>

## ls -al

디렉토리 내부의 파일 리스트를 보기 위한 명령어.

윈도우탐색기를 사용하여 파일 존재를 확인한다.


<br>

## wget

http를 사용하여 파일을 다운로드 받는 명령어이다.

대상 url을 카피하여 웹브라우저에서 수작업으로 직접 다운로드 받는다.


<br>

## tar or unzip

압축을 풀기위한 명령어이다.

보통 tar.gz 이나 zip의 확장자를 갖는다.

윈도우즈에 설치된 알집이나 반디집같은 압축 프로그램을 사용하여 수동으로 직접 압축을 풀어준다.


<br>

## git

git 리파지터리의 프로젝트 전체를 다운로드 받는 명령어이다.

github.com의 프로젝트로 이동하여 'Goto File', 'Add File' 옆의 'Code' > Download ZIP을 클릭하여 프로젝트 전체가 압축된 파일을 다운로드 받고, 압축을 푼다.

<br>

혹은 아나콘다 파워셀에서 다음과 같이 git을 설치하고 리눅스 명령어와 동일한 방법으로 실행시킬 수 있다.
```
conda install git
```


## nvidia-smi

GPU 사용상태를 보기 위한 명령어.

따로 설치하지 않아도 된다.

아나콘다 파워셀에서 똑같은 방법으로 실행 시킬 수 있다.






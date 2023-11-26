<a name="readme-top"></a>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/FELAB-KHU/SNPQuant">
    <img src="images/Felab_logo 2.png" alt="Logo" width="520" height="100">
  </a>

  <h3 align="center">퀀트 투자 모델의 멀티모달리티 적용</h3>

  <p align="center">
    최신 데이터 과학 기술을 활용한 혁신적인 투자 전략 개발 프로젝트
    <br />
    <a href="https://github.com/FELAB-KHU/SNPQuant"><strong>문서 탐색하기 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/FELAB-KHU/SNPQuant">데모 보기</a>
    ·
    <a href="https://github.com/FELAB-KHU/SNPQuant/issues">버그 신고</a>
    ·
    <a href="https://github.com/FELAB-KHU/SNPQuant/issues">기능 요청</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![(Re-)Imag(in)ing Price Trends][product-screenshot]](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13268)
[![Product Name Screen Shot][product-screenshot2]](https://github.com/AI4Finance-Foundation/FinGPT)

현대 투자의 세계는 데이터의 다양성과 복잡성으로 인해 계속해서 진화하고 있습니다. 이러한 환경에서, "퀀트 투자 모델의 멀티모달리티 적용" 프로젝트는 혁신적인 접근 방식을 통해 금융 시장의 동향을 예측하고 분석합니다. 이 프로젝트는 Python, Docker, PyTorch, Git, GitHub 등 최첨단 기술을 활용하여 구축되었습니다.

프로젝트의 핵심은 다양한 데이터 소스와 API의 통합입니다. 여기에는 YouTube Data v3 API, OpenAI Whisper 같은 STT (Speech-to-Text) 기술, FinGPT (Llama2), Reddit API, S&P Data를 활용한 NLP (Natural Language Processing) 접근 방식, 그리고 S&P Data와 yfinance를 통한 정량적 데이터 분석이 포함됩니다.

이 멀티모달 데이터 접근 방식을 통해, 우리는 금융 데이터의 다양한 측면을 분석하고, 보다 정확하고 효과적인 투자 결정을 내리는 데 필요한 통찰력을 제공합니다. 이 프로젝트는 데이터 과학과 퀀트 금융 분야에서 새로운 지평을 여는 것을 목표로 하고 있습니다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

이 프로젝트는 다음과 같은 도구 및 기술을 사용하여 구축되었습니다.

* [![Python][Python-shield]][Python-url]
* [![Docker][Docker-shield]][Docker-url]
* [![PyTorch][PyTorch-shield]][PyTorch-url]
* [![Git][Git-shield]][Git-url]
* [![Github][Github-shield]][Github-url]


### 데이터 소스

프로젝트에서는 다음과 같은 다양한 데이터 소스 및 API를 사용하였습니다.

#### STT
* YouTube Data v3 API
* OpenAI Whisper
* FinGPT (Llama2)

#### NLP
* Reddit API
* S&P Data

#### Quantitive
* S&P Data
* yfinance


<!-- GETTING STARTED -->
## Getting Started

다음은 로컬에서 프로젝트를 설정하는 방법을 설명하는 예시입니다.
로컬 사본을 설정하고 실행하려면 다음의 간단한 예제 단계를 따르세요.

### Prerequisites

다음은 소프트웨어를 사용하는 데 필요한 항목을 나열하는 방법과 설치 방법의 예입니다.
* Python
* CUDA (Example)

  ```sh
  nvidia-smi
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 536.40                 Driver Version: 536.40       CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce RTX 4070 Ti   WDDM  | 00000000:01:00.0  On |                  N/A |
    |  0%   43C    P8              13W / 285W |   1295MiB / 12282MiB |     24%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
  ```

### Installation

_아래는 사용자들이 앱을 설치하고 설정하는 방법을 안내하는 예시입니다. 이 템플릿은 외부 의존성이나 서비스에 의존하지 않습니다._

1. OPEN AI API 키와 YouTube Data v3 API 키를 별도로 받으세요. OPEN AI API는 유료입니다.
2. 저장소를 클론하세요:
   ```sh
   git clone https://github.com/FELAB/SNPQuant.git
   ```
3. pipenv를 사용하여 필요한 패키지들을 설치하세요:
   ```sh
   pipenv install
   ```
4. `config.py`에 API 키를 입력하세요:
   ```py
   OPENAI_API_KEY = '여기에 OPEN AI API 키를 입력';
   YOUTUBE_API_KEY = '여기에 YouTube Data v3 API 키를 입력';
   ```

주의: OPEN AI API와 YouTube Data v3 API를 사용하기 위해서는 각각의 공식 웹사이트에서 API 키를 직접 발급받아 사용해야 합니다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

이 섹션에서는 프로젝트를 통해 어떻게 혁신적인 퀀트 투자 모델을 구축하고 활용할 수 있는지에 대한 유용한 예시를 보여줍니다. 여기에는 멀티모달 데이터 분석, STT와 NLP를 활용한 시장 동향 예측, 그리고 yfinance를 통한 실시간 금융 데이터 분석의 스크린샷과 코드 예시를 포함할 수 있습니다. 또한, 이 프로젝트에 대한 보다 자세한 정보와 추가 리소스에 대한 링크를 제공할 수 있습니다.

_더 많은 예시와 자세한 정보는 [공식 문서](https://example.com)에서 확인하세요._

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] 멀티모달 데이터 처리 방법 개발
- [x] 기술 스택 및 API 연동
- [ ] 추가 데이터 소스 통합 및 분석 예시 추가
- [ ] 사용자 매뉴얼 및 문서 개선
- [ ] 다양한 언어 지원 확대
    - [ ] 한국어
    - [ ] 영어

현재 진행 중이거나 제안된 기능 및 알려진 이슈에 대한 전체 목록은 [공개 이슈](https://github.com/othneildrew/Best-README-Template/issues)에서 확인할 수 있습니다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

기여는 오픈소스 커뮤니티를 배우고, 영감을 얻고, 창조할 수 있는 놀라운 공간으로 만들어 줍니다. 여러분의 모든 기여는 **대단히 감사드리겠습니다**.

이 기능을 개선할 수 있는 제안이 있다면 리포지토리를 포크하고 풀 리퀘스트를 만들어 주세요. "개선" 태그와 함께 이슈를 간단히 열 수도 있습니다.
프로젝트에 별표를 주는 것을 잊지 마세요! 다시 한번 감사드립니다!

1. 프로젝트 포크
2. 피처 브랜치를 생성 (`git checkout -b feature/AmazingFeature`).
3. 변경 사항을 커밋 (`git commit -m 'Add some AmazingFeature'`).
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. 풀 리퀘스트 요청

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE.txt`를 참조하세요.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Inyeol-Choi - [@Linkedin](https://www.linkedin.com/in/in-yeol-choi-98b21b26b/) - chldlsel@khu.ac.kr

Project Link: [FELAB/SNPQuant](https://github.com/FELAB-KHU/SNPQuant)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

이 공간에 도움이 되었거나 공로를 인정하고 싶은 리소스를 나열하세요. 시작을 위해 제가 가장 좋아하는 몇 가지를 포함했습니다!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/in-yeol-choi-98b21b26b/
[product-screenshot]: images/trend.png
[product-screenshot2]: images/logo_transparent_background.png

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white
[Python-url]: https://python.org

[Git-shield]: https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white
[Git-url]: https://git-scm.com

[Github-shield]: https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white
[Github-url]: https://github.com

[PyTorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org

[Docker-shield]: https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=Docker&logoColor=white
[Docker-url]: https://docker.com





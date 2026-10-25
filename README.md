# PathLight GI — iluminação global estilo path tracing para o Dusklight

Mod nativo para o [Dusklight](https://github.com/TwilitRealm/dusklight) que adiciona iluminação global (GI)
em tempo real, inspirada em path tracing: raios Monte Carlo lançados no espaço da tela, múltiplos
rebotes de luz, sangramento de cor entre superfícies, luz do céu, sombras de contato, acumulação
temporal e denoiser que respeita bordas.

> **Importante:** não é path tracing "de verdade" (como o do RTX). O Dusklight roda sobre a geometria
> original do jogo, sem BVH/RT cores. O mod traça os raios contra o **buffer de profundidade** (SSGI).
> Ele não enxerga o que está fora da tela ou atrás de objetos — é a mesma limitação de qualquer SSGI.

## Instalação (jogador)

1. Baixe o arquivo `pathlight_gi.dusk` (veja "Como obter o .dusk para o seu sistema" abaixo).
2. Copie para a pasta de mods:
   - Windows: `%APPDATA%\TwilitRealm\Dusklight\mods`
   - Linux: `~/.local/share/TwilitRealm/Dusklight/mods`
   - macOS: `~/Library/Application Support/TwilitRealm/Dusklight/mods`
3. Abra o Dusklight e ative **PathLight GI** no menu de Mods. Os ajustes (qualidade, intensidade do
   rebote, oclusão, alcance, luz do céu, etc.) ficam no painel do mod e valem em tempo real.

## Como obter o .dusk para o seu sistema

Um mod nativo precisa ser compilado para cada plataforma. O arquivo `pathlight_gi.dusk` deste pacote
foi compilado **apenas para Linux x86_64**. Para Windows/macOS/Android use o CI incluído:

1. Crie um repositório no GitHub e envie o conteúdo desta pasta.
2. A aba **Actions** compila para Linux, macOS, Windows, iOS e Android e junta tudo num único
   `.dusk` multiplataforma (artefato `mod-combined`).
3. Para publicar uma release com o arquivo anexado: `git tag v1.0.0 && git push --tags`.

## Compilar localmente

```
cmake -B build -G Ninja
cmake --build build
# resultado: build/mods/pathlight_gi.dusk
```

Requer CMake 3.26+, Ninja e um compilador C++20. O SDK do Dusklight (v2.0.3) é baixado automaticamente;
use `-DDUSKLIGHT_DIR=/caminho/do/dusklight` para usar um checkout existente.

## Estrutura

- `src/mod.cpp` — integração com o renderer (hooks de estágio, passes de compute, composição, UI).
- `res/*.wgsl` — shaders: `trace` (raios), `temporal` (acumulação), `denoise` (à-trous), `composite`.
- `tools/harness/` — banco de testes WebGPU em Chromium que roda os mesmos shaders numa cena
  Cornell box (`node tools/harness/run.js <pasta-saida>`).

## Estado dos testes

- Os shaders foram validados no harness WebGPU (Cornell box: sangramento de cor vermelho/verde,
  oclusão, convergência temporal).
- O código C++ compila e empacota contra o SDK.
- **Não foi testado dentro do jogo**, pois isso exige uma cópia do disco original e uma GPU. Se algo
  falhar, o log do Dusklight mostra mensagens com o prefixo `dev.community.pathlight_gi`.
